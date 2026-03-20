import math
import random
import time
from collections import defaultdict

# Alpha-style MCTS (PUCT) that uses a network providing (policy_probs, value)
# The network must implement a `predict(state)` method returning (policy_logits_tensor, value_float)
# or the caller can pass a function `evaluator(state)` returning (policy_probs, value)

class AlphaNode:
    def __init__(self, state, parent=None, move=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def uct_score(self, child, c_puct):
        # PUCT: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        return child.q_value() + c_puct * child.prior * math.sqrt(self.visit_count + 1e-9) / (1 + child.visit_count)


def select(node, c_puct=1.0):
    # descend until leaf (no children)
    while node.children:
        # pick child maximizing PUCT
        node = max(node.children.values(), key=lambda ch: node.uct_score(ch, c_puct))
    return node


def expand_with_policy(node, evaluator):
    # evaluator(state) -> (policy_probs: dict move_str->prob, value: float)
    if node.state.is_terminal():
        return node, 1.0 if node.state.result() == 1 else -1.0 if node.state.result() == -1 else 0.0

    policy_probs, value = evaluator(node.state)
    # policy_probs: dict mapping move_str to prior probability
    for move_str, prob in policy_probs.items():
        # create child state by cloning and playing the move
        # move_str must be parseable by Move.from_str in calling code; here we rely on legal_moves strings
        s2 = node.state.clone()
        # parse move
        from Jeu.Yolah import Move  # local import to avoid cycle at top
        m = Move.from_str(move_str)
        s2.play(m)
        node.children[move_str] = AlphaNode(s2, parent=node, move=m, prior=prob)

    # pick a random child to continue the search (common approach) - but we'll return node for evaluation externally
    return node, value


def backpropagate(node, value):
    # value is from the perspective of the current node's player
    while node:
        node.visit_count += 1
        node.value_sum += value
        value = -value
        node = node.parent


def alpha_mcts(root_state, evaluator, iterations=800, time_limit_s=None, c_puct=1.0, verbose=True):
    from Jeu.Yolah import Move
    root = AlphaNode(root_state)
    start = time.perf_counter()
    it = 0
    while True:
        if time_limit_s and time.perf_counter() - start > time_limit_s:
            break
        if not time_limit_s and it >= iterations:
            break

        leaf = select(root, c_puct=c_puct)
        if leaf.state.is_terminal():
            # terminal node -> backprop the result
            result = leaf.state.result()
            value = result if leaf.state.current_player() == 0 else -result
            backpropagate(leaf, value)
            it += 1
            continue

        # expand leaf using network policy and get value estimate
        node_after_expand, value = expand_with_policy(leaf, evaluator)
        # If expansion generated children, pick one child for backup (could also evaluate at leaf)
        if node_after_expand.children:
            # choose child with highest prior (or random)
            child = max(node_after_expand.children.values(), key=lambda ch: ch.prior)
            # Use network value for backpropagation
            backpropagate(child, value)
        else:
            # no children (shouldn't happen), use value at leaf
            backpropagate(leaf, value)

        it += 1

    # build stats similar to original mcts_collect_stats: move_str -> (visits, value_sum)
    stats = {}
    for move_str, ch in root.children.items():
        stats[str(move_str)] = (ch.visit_count, ch.value_sum)

    elapsed = time.perf_counter() - start
    if verbose:
        print(f"AlphaMCTS: iterations={it}, elapsed={elapsed:.2f}s")

    if not stats:
        return Move.none(), stats

    best_move_str = max(stats.items(), key=lambda kv: kv[1][0])[0]
    best_move = Move.from_str(best_move_str)
    return best_move, stats
