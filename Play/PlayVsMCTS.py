import sys
from pathlib import Path
from typing_extensions import final
sys.path.append(str(Path(__file__).resolve().parents[1]))

from multiprocessing import Pool, cpu_count
import argparse
import os
import torch

try:
    from net.nn_model import YolahNet
    from net.evaluator import NeuralEvaluator
    from MCTS.alpha_mcts import alpha_mcts
except Exception:
    YolahNet = None
    NeuralEvaluator = None
    alpha_mcts = None
from Jeu.Yolah import Yolah, Move
from Jeu.YolahInterface import YolahState
from MCTS.MCTS import mcts_collect_stats

# --- Helpers de (dé)sérialisation d'état ---

def serialize_state(state: YolahState):
    # tuple d'entiers + ply => parfaitement sérialisable
    return state.game.get_state()

def deserialize_state(state_tuple):
    g = Yolah()
    g.black, g.white, g.empty, g.black_score, g.white_score, g.ply = state_tuple
    return YolahState(g)

# --- Worker fonction (doit être top-level pour multiprocessing) ---

def _worker_collect_stats(state_tuple, iterations, time_limit_s):
    s = deserialize_state(state_tuple)
    return mcts_collect_stats(s, iterations=iterations, time_limit_s=time_limit_s)

# --- Merge des stats ---

def merge_stats(list_of_dicts):
    merged = {}  # move_str -> [total_visits, total_value_sum]

    for d in list_of_dicts:
        for move_str, (visits, value_sum) in d.items():
            if move_str not in merged:
                merged[move_str] = [0, 0.0]
            merged[move_str][0] += visits
            merged[move_str][1] += value_sum

    # Convertir en (visits, value)
    final = {}
    for move_str, (v, s) in merged.items():
        final[move_str] = (v, s / v if v > 0 else 0)

    return final

def parallel_mcts(root_state, iterations=800, time_limit_s=None, workers=None):
    if workers is None:
        workers = cpu_count()

    state_tuple = serialize_state(root_state)
    tasks = [(state_tuple, iterations, time_limit_s) for _ in range(workers)]

    with Pool(processes=workers) as p:
        results = p.starmap(_worker_collect_stats, tasks)

    # MERGE CORRIGÉ
    merged = merge_stats(results)

    if not merged:
        return Move.none(), merged

    # CHOIX DU MEILLEUR COUP
    best_move_str = max(merged.items(), key=lambda kv: kv[1][0])[0]
    best_move = Move.from_str(best_move_str)

    return best_move, merged


def load_evaluator(model_path: str = None, device: str = "cpu"):
    """Load a PyTorch model and return a NeuralEvaluator or None if not available."""
    if model_path is None:
        return None
    if NeuralEvaluator is None or YolahNet is None:
        print("Neural evaluator or model not available (missing imports)")
        return None
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    model = YolahNet()
    try:
        ck = torch.load(model_path, map_location=device)
        if isinstance(ck, dict) and "model_state" in ck:
            model.load_state_dict(ck["model_state"])
        else:
            model.load_state_dict(ck)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    return NeuralEvaluator(model, device=device)

# --- Boucle de jeu ---

def play_human_vs_mcts(evaluator=None, use_net=False, iterations=800, time_limit_s=None):
    state = YolahState()
    print(state.game)

    while not state.is_terminal():
        if state.current_player() == 0:
            # Humain (Noir)
            moves = state.legal_moves()
            print("Coups possibles :", ", ".join(str(m) for m in moves))
            mv = input("Ton coup : ").strip()
            state.play(Move.from_str(mv))
        else:
            if use_net and evaluator is not None and alpha_mcts is not None:
                print("IA réseau+MCTS réfléchit...")
                move, merged = alpha_mcts(state, evaluator, iterations=iterations, time_limit_s=time_limit_s)
                print("IA (network) joue:", move)
                state.play(move)
            else:
                print("IA MCTS parallèle réfléchit...")
                move, merged = parallel_mcts(state, time_limit_s=10, workers=None)
                print("IA joue :", move)
                state.play(move)

        print(state.game)

    print("Partie terminée.")
    print("Résultat :", state.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-net", action="store_true", help="Use neural network guided MCTS")
    parser.add_argument("--model", type=str, default=None, help="Path to model .pt file")
    parser.add_argument("--iterations", type=int, default=800, help="MCTS iterations when using network")
    parser.add_argument("--time", type=float, default=None, help="Time limit seconds when using network")
    args = parser.parse_args()

    evaluator = None
    if args.use_net:
        evaluator = load_evaluator(args.model)
        if evaluator is None:
            print("Falling back to classical MCTS because evaluator could not be loaded.")
            args.use_net = False

    play_human_vs_mcts(evaluator=evaluator, use_net=args.use_net, iterations=args.iterations, time_limit_s=args.time)