import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from net.nn_model import YolahNet
from net.evaluator import NeuralEvaluator
from MCTS.alpha_mcts import alpha_mcts
from Jeu.YolahInterface import YolahState

import torch


def smoke_test():
    # create model and evaluator
    model = YolahNet()
    evaluator = NeuralEvaluator(model)

    state = YolahState()
    print('Initial state:')
    print(state.game)

    # run a short MCTS guided by the (randomly initialized) network
    move, stats = alpha_mcts(state, evaluator, iterations=50, verbose=True)
    print('Selected move by AlphaMCTS:', move)
    print('Stats (top 10):')
    for i, (m, (vis, val)) in enumerate(sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True)[:10]):
        print(f"{m}: visits={vis}, value_sum={val}")


if __name__ == '__main__':
    smoke_test()
