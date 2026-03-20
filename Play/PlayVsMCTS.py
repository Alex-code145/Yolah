import sys
from pathlib import Path
from typing_extensions import final
sys.path.append(str(Path(__file__).resolve().parents[1]))

from multiprocessing import Pool, cpu_count
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

# --- Boucle de jeu ---

def play_human_vs_mcts():
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
            print("IA MCTS parallèle réfléchit...")
            move, merged = parallel_mcts(state, time_limit_s=3, workers=None)
            print("IA joue :", move)
            state.play(move)

        print(state.game)

    print("Partie terminée.")
    print("Résultat :", state.result())


if __name__ == "__main__":
    play_human_vs_mcts()