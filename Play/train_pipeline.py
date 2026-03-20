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
):
    model.to(device)
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
                for xb, pi_b, z_b in dl:
                    xb = xb.to(device)
                    pi_b = pi_b.to(device)
                    z_b = z_b.to(device)
                    logits, v = model(xb)
                    # policy loss: cross-entropy with pi_b
                    logp = F.log_softmax(logits, dim=1)
                    policy_loss = - (pi_b * logp).sum(dim=1).mean()
                    value_loss = F.mse_loss(v, z_b)
                    loss = policy_loss + value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # save checkpoint
        torch.save({"model_state": model.state_dict(), "iter": it}, save_path)
        t1 = time.time()
        print(f"Iter {it}/{num_iters}: games={games_per_iter}, replay={len(replay)}, avg_result={sum(results)/len(results):.3f}, wall={t1-t0:.2f}s")


if __name__ == "__main__":
    # small smoke training run to validate the pipeline
    model = YolahNet()
    # smoke: use network vs classical MCTS opponent to collect diversified data
    train_loop(model, device="cpu", num_iters=20, games_per_iter=20, mcts_iters=20, opponent="mcts", opponent_mcts_iters=40, buffer_size=1000, batch_size=8, epochs=1, save_path="model.pt")
