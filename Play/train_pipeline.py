import random
import time
from collections import deque
from typing import Deque, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from net.nn_model import YolahNet, state_to_tensor, OUTPUT_POLICY_SIZE, move_to_index
from net.evaluator import NeuralEvaluator
from MCTS.alpha_mcts import alpha_mcts
from MCTS.MCTS import mcts_collect_stats
from Jeu.YolahInterface import YolahState
from Jeu.Yolah import Yolah, Move


class ReplaySample:
    def __init__(self, state_tuple, pi: Dict[str, float], z: float):
        self.state = state_tuple  # tuple from Yolah.get_state()
        self.pi = pi  # move_str -> prob
        self.z = z  # final game result: +1 black win, -1 white, 0 draw


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer: Deque[ReplaySample] = deque(maxlen=capacity)

    def push(self, sample: ReplaySample):
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[ReplaySample]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class PolicyValueDataset(Dataset):
    def __init__(self, samples: List[ReplaySample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        black, white, empty, black_score, white_score, ply = s.state
        x = state_to_tensor(black, white, empty, ply & 1)
        # policy target vector
        pi_vec = torch.zeros(OUTPUT_POLICY_SIZE, dtype=torch.float32)
        for mstr, p in s.pi.items():
            try:
                idx = move_to_index(Move.from_str(mstr))
                pi_vec[idx] = float(p)
            except Exception:
                pass
        # value target from perspective of player to move in that state
        player = Yolah.WHITE_PLAYER if ply & 1 else Yolah.BLACK_PLAYER
        z = float(s.z)
        value_target = z if player == Yolah.BLACK_PLAYER else -z
        return x, pi_vec, torch.tensor(value_target, dtype=torch.float32)


def self_play_game(evaluator: NeuralEvaluator, mcts_iters=200, temp_threshold=10, time_limit_s=None, opponent: str = "self", opponent_mcts_iters: int = 400):
    state = YolahState()
    samples = []

    while not state.is_terminal():
        # Decide which engine plays this ply
        current = state.current_player()
        if opponent == "mcts" and current == Yolah.WHITE_PLAYER:
            # Classical MCTS (opponent)
            stats = mcts_collect_stats(state, iterations=opponent_mcts_iters, time_limit_s=time_limit_s, verbose=False)
            # select move according to stats
            if not stats:
                moves = state.legal_moves()
                chosen = random.choice(moves)
                state.play(chosen)
                continue
            # build visits dict similar to alpha branch
            visits = {m: v for m, (v, vs) in stats.items()}
            total = sum(visits.values())
            if total == 0:
                legal = state.legal_moves()
                pi = {str(m): 1.0 / len(legal) for m in legal}
            else:
                pi = {m: visits[m] / total for m in visits}
            # record sample (we still store positions for training)
            samples.append((state.game.get_state(), pi))

            # choose move (deterministic argmax for opponent)
            chosen_move = Move.from_str(max(pi.items(), key=lambda kv: kv[1])[0])
            state.play(chosen_move)
            continue

        # run alpha mcts to obtain stats (visits per move)
        move, stats = alpha_mcts(state, evaluator, iterations=mcts_iters, time_limit_s=time_limit_s, verbose=False)
        # stats: move_str -> (visits, value_sum)
        if not stats:
            # fallback: choose random legal move
            moves = state.legal_moves()
            chosen = random.choice(moves)
            state.play(chosen)
            continue

        # build pi from visits
        visits = {m: v for m, (v, vs) in stats.items()}
        total = sum(visits.values())
        if total == 0:
            # uniform
            legal = state.legal_moves()
            pi = {str(m): 1.0 / len(legal) for m in legal}
        else:
            pi = {m: visits[m] / total for m in visits}

        # record sample
        samples.append((state.game.get_state(), pi))

        # choose move: sample if ply < temp_threshold else choose argmax
        ply = state.game.ply
        if ply < temp_threshold:
            # sample according to pi
            moves_list = list(pi.items())
            ms, ps = zip(*moves_list)
            chosen_str = random.choices(ms, weights=ps, k=1)[0]
            chosen_move = Move.from_str(chosen_str)
        else:
            # argmax
            chosen_move = Move.from_str(max(pi.items(), key=lambda kv: kv[1])[0])

        state.play(chosen_move)

    # game finished
    result = state.result()  # +1 black, -1 white, 0 draw
    replay_samples = []
    for st, pi in samples:
        replay_samples.append(ReplaySample(st, pi, result))
    return replay_samples, result


def train_loop(
    model: YolahNet,
    device: str = "cpu",
    num_iters: int = 100,
    games_per_iter: int = 10,
    mcts_iters: int = 200,
    opponent: str = "self",
    opponent_mcts_iters: int = 400,
    buffer_size: int = 10000,
    batch_size: int = 64,
    epochs: int = 1,
    save_path: str = "model.pt",
    lr: float = 1e-3,
    eval_games: int = 20,
    eval_mcts_iters: int = 800,
    eval_opponent_mcts_iters: int = 800,
    log_csv: str = "training_log.csv",
    checkpoint_prefix: str = "model_iter",
):
    model.to(device)
    # prefer OpenVINO evaluator when available (inference-accelerated)
    try:
        from net.ov_inference import export_model_to_onnx, OpenVINOEvaluator
        ov_available = True
    except Exception:
        ov_available = False

    if ov_available:
        onnx_path = "model_openvino.onnx"
        try:
            export_model_to_onnx(model, onnx_path)
            evaluator = OpenVINOEvaluator(onnx_path, device="CPU")
            print("Using OpenVINO evaluator for MCTS inference.")
        except Exception as e:
            print(f"OpenVINO evaluator setup failed: {e}; falling back to NeuralEvaluator")
            evaluator = NeuralEvaluator(model, device=device)
    else:
        evaluator = NeuralEvaluator(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=buffer_size)

    for it in range(1, num_iters + 1):
        t0 = time.time()
        # self-play games
        results = []
        for g in range(games_per_iter):
            samples, result = self_play_game(evaluator, mcts_iters=mcts_iters, temp_threshold=10, opponent=opponent, opponent_mcts_iters=opponent_mcts_iters)
            for s in samples:
                replay.push(s)
            results.append(result)

        # training
        if len(replay) >= batch_size:
            for ep in range(epochs):
                # create dataset from random samples
                batch_samples = replay.sample(batch_size)
                dataset = PolicyValueDataset(batch_samples)
                dl = DataLoader(dataset, batch_size=min(32, batch_size), shuffle=True)
                model.train()
                # accumulate losses/metrics for logging
                policy_loss_sum = 0.0
                value_loss_sum = 0.0
                entropy_sum = 0.0
                batches = 0
                for xb, pi_b, z_b in dl:
                    xb = xb.to(device)
                    pi_b = pi_b.to(device)
                    z_b = z_b.to(device)
                    logits, v = model(xb)
                    # policy loss: cross-entropy with pi_b
                    logp = F.log_softmax(logits, dim=1)
                    probs = F.softmax(logits, dim=1)
                    # small epsilon for numeric stability
                    eps = 1e-8
                    policy_loss = - (pi_b * logp).sum(dim=1).mean()
                    value_loss = F.mse_loss(v, z_b)
                    loss = policy_loss + value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # metrics
                    batches += 1
                    policy_loss_sum += float(policy_loss.detach().cpu().numpy())
                    value_loss_sum += float(value_loss.detach().cpu().numpy())
                    # entropy per sample
                    ent = - (probs * (torch.log(probs + eps))).sum(dim=1).mean()
                    entropy_sum += float(ent.detach().cpu().numpy())

                # average over batches
                if batches > 0:
                    avg_policy_loss = policy_loss_sum / batches
                    avg_value_loss = value_loss_sum / batches
                    avg_entropy = entropy_sum / batches
                else:
                    avg_policy_loss = avg_value_loss = avg_entropy = 0.0

        # save checkpoint (incremental)
        ck_path = f"{checkpoint_prefix}_{it}.pt"
        torch.save({"model_state": model.state_dict(), "iter": it}, ck_path)

        # Evaluation vs classical MCTS opponent (periodic)
        eval_win = eval_draw = eval_loss = 0
        eval_games_played = 0
        if eval_games > 0:
            # build evaluator for current model
            try:
                if ov_available:
                    export_model_to_onnx(model, onnx_path)
                    eval_evaluator = OpenVINOEvaluator(onnx_path, device="CPU")
                else:
                    eval_evaluator = NeuralEvaluator(model, device=device)
            except Exception:
                eval_evaluator = NeuralEvaluator(model, device=device)
            for gidx in range(eval_games):
                # alternate starting player: even -> model as Black, odd -> model as White
                model_black = (gidx % 2 == 0)
                # play a single game
                s = YolahState()
                while not s.is_terminal():
                    if model_black:
                        # model plays Black when current_player==BLACK_PLAYER else opponent
                        if s.current_player() == Yolah.BLACK_PLAYER:
                            mv, _ = alpha_mcts(s, eval_evaluator, iterations=eval_mcts_iters, time_limit_s=None, verbose=False)
                            s.play(mv)
                        else:
                            stats = mcts_collect_stats(s, iterations=eval_opponent_mcts_iters, time_limit_s=None, verbose=False)
                            if not stats:
                                s.play(Move.none())
                            else:
                                best = max(stats.items(), key=lambda kv: kv[1][0])[0]
                                s.play(Move.from_str(best))
                    else:
                        # model plays White
                        if s.current_player() == Yolah.WHITE_PLAYER:
                            mv, _ = alpha_mcts(s, eval_evaluator, iterations=eval_mcts_iters, time_limit_s=None, verbose=False)
                            s.play(mv)
                        else:
                            stats = mcts_collect_stats(s, iterations=eval_opponent_mcts_iters, time_limit_s=None, verbose=False)
                            if not stats:
                                s.play(Move.none())
                            else:
                                best = max(stats.items(), key=lambda kv: kv[1][0])[0]
                                s.play(Move.from_str(best))

                res = s.result()
                # from perspective of Black: +1 black wins, -1 white wins
                if res == 1:
                    # black won
                    if model_black:
                        eval_win += 1
                    else:
                        eval_loss += 1
                elif res == -1:
                    # white won
                    if not model_black:
                        eval_win += 1
                    else:
                        eval_loss += 1
                else:
                    eval_draw += 1
                eval_games_played += 1

        t1 = time.time()

        # logging: print summary
        avg_result = sum(results) / len(results) if results else 0.0
        print(f"Iter {it}/{num_iters}: games={games_per_iter}, replay={len(replay)}, avg_result={avg_result:.3f}, wall={t1-t0:.2f}s")
        # print metrics if available
        if 'avg_policy_loss' in locals():
            print(f"  train: policy_loss={avg_policy_loss:.4f}, value_loss={avg_value_loss:.4f}, policy_entropy={avg_entropy:.4f}")
        if eval_games_played > 0:
            winrate = eval_win / eval_games_played
            print(f"  eval ({eval_games_played} games) win/draw/loss = {eval_win}/{eval_draw}/{eval_loss} (winrate={winrate:.3f})")

        # append CSV log
        try:
            header = False
            csv_line = f"{it},{games_per_iter},{len(replay)},{avg_result:.3f},{avg_policy_loss if 'avg_policy_loss' in locals() else 0.0:.6f},{avg_value_loss if 'avg_value_loss' in locals() else 0.0:.6f},{avg_entropy if 'avg_entropy' in locals() else 0.0:.6f},{eval_games_played},{eval_win},{eval_draw},{eval_loss}\n"
            if not Path(log_csv).exists():
                with open(log_csv, 'w') as f:
                    f.write("iter,games,replay,avg_result,policy_loss,value_loss,policy_entropy,eval_games,eval_win,eval_draw,eval_loss\n")
            with open(log_csv, 'a') as f:
                f.write(csv_line)
        except Exception as e:
            print(f"Failed to write CSV log: {e}")


if __name__ == "__main__":
    # short diagnostic training run with higher MCTS sims to produce sharper targets
    # Keep the run small so it completes quickly while showing effect of more sims.
    # Tune thread usage for CPU (oneDNN/MKL) to match machine cores.
    try:
        import os
        import multiprocessing
        ncpu = max(1, int(os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count())))
        torch.set_num_threads(ncpu)
        torch.set_num_interop_threads(ncpu)
        print(f"Configured torch threads={ncpu}")
    except Exception:
        pass

    # Try to detect Intel Extension for PyTorch (IPEX) to accelerate CPU training on Intel hardware.
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
        ipex_available = True
        print("Intel Extension for PyTorch (IPEX) detected — will optimize model for CPU if used.")
    except Exception:
        ipex_available = False
        print("IPEX not available; running standard PyTorch on CPU. To use Intel GPU/accelerations, install IPEX or OpenVINO as described in the README.")

    model = YolahNet()
    if ipex_available:
        try:
            # ipex.optimize returns an optimized model; keep a reference to it.
            model = ipex.optimize(model)
            print("Model optimized with IPEX.")
        except Exception as e:
            print("IPEX optimization failed, continuing without it:", e)

    train_loop(
        model,
        device="cpu",
        num_iters=3,                # few training iterations
        games_per_iter=10,          # games per iteration
        mcts_iters=100,            # increased MCTS sims to sharpen pi targets
        opponent="mcts",
        opponent_mcts_iters=10,    # lighter opponent MCTS for speed
        buffer_size=500,
        batch_size=64,
        epochs=2,                  # a couple epochs per iteration
        eval_games=10,
        eval_mcts_iters=50,
        save_path="model.pt",
    )


def overfit_test(num_samples: int = 50, epochs: int = 200, lr: float = 1e-3, device: str = "cpu"):
    """Create a small synthetic dataset and train the model to verify it can fit data.

    We generate random legal positions by playing random moves from the initial state.
    For each position we pick one legal move as the target (one-hot policy) and a random game result z in {-1,1}.
    Then we train for many epochs and print loss trace. This checks the model & loss plumbing.
    """
    print("Running overfit test...")
    model = YolahNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Build samples
    samples = []
    for i in range(num_samples):
        s = YolahState()
        # play a random number of random moves to diversify positions
        steps = random.randint(0, 6)
        for _ in range(steps):
            moves = s.legal_moves()
            if moves == [Move.none()]:
                s.play(Move.none())
            else:
                s.play(random.choice(moves))

        legal = s.legal_moves()
        if not legal:
            pi = {str(Move.none()): 1.0}
        else:
            chosen = random.choice(legal)
            pi = {str(m): 0.0 for m in legal}
            pi[str(chosen)] = 1.0

        z = random.choice([1.0, -1.0])
        samples.append(ReplaySample(s.game.get_state(), pi, z))

    dataset = PolicyValueDataset(samples)
    dl = DataLoader(dataset, batch_size=min(16, num_samples), shuffle=True)

    for ep in range(1, epochs + 1):
        model.train()
        tot_pl = 0.0
        tot_vl = 0.0
        batches = 0
        for xb, pi_b, z_b in dl:
            xb = xb.to(device)
            pi_b = pi_b.to(device)
            z_b = z_b.to(device)
            logits, v = model(xb)
            logp = F.log_softmax(logits, dim=1)
            policy_loss = - (pi_b * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(v, z_b)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_pl += float(policy_loss.detach().cpu().numpy())
            tot_vl += float(value_loss.detach().cpu().numpy())
            batches += 1

        if batches > 0:
            print(f"Overfit epoch {ep}/{epochs}: policy_loss={tot_pl/batches:.6f}, value_loss={tot_vl/batches:.6f}")

    print("Overfit test done. Saving small overfit model to overfit_model.pt")
    torch.save({"model_state": model.state_dict()}, "overfit_model.pt")
