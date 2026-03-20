import math
import random
import time

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def uct(parent, child, c=1.4):
    # child.value() is from the child player's perspective; negate for parent perspective.
    return -child.value() + c * math.sqrt(
        math.log(parent.visit_count + 1) / (child.visit_count + 1e-9)
    )


def rollout(state):
    s = state.clone()
    while not s.is_terminal():
        moves = s.legal_moves()
        s.play(random.choice(moves))
    return s.result()


def select(node):
    while node.children:
        node = max(node.children, key=lambda ch: uct(node, ch))
    return node


def expand(node):
    if node.state.is_terminal():
        return node
    moves = node.state.legal_moves()
    for m in moves:
        s2 = node.state.clone()
        s2.play(m)
        node.children.append(MCTSNode(s2, parent=node, move=m))
    return random.choice(node.children)


def backpropagate(node, reward):
    while node:
        node.visit_count += 1
        node.value_sum += reward
        reward = -reward
        node = node.parent


def mcts_collect_stats(root_state, iterations=800, time_limit_s=None, verbose=True):
    root = MCTSNode(root_state)
    start = time.perf_counter()
    it = 0

    while True:
        if time_limit_s and time.perf_counter() - start > time_limit_s:
            break
        if not time_limit_s and it >= iterations:
            break

        leaf = expand(select(root))
        # `result` is absolute (+1 black wins, -1 white wins). Convert to leaf player's perspective.
        result = rollout(leaf.state)
        reward = result if leaf.state.current_player() == 0 else -result
        backpropagate(leaf, reward)
        it += 1

    # Retourne visits + value_sum (PAS value)
    stats = {}
    for ch in root.children:
        stats[str(ch.move)] = (ch.visit_count, ch.value_sum)

    elapsed = time.perf_counter() - start
    if verbose:
        print(f"MCTS: iterations={it}, elapsed={elapsed:.2f}s")

    return stats