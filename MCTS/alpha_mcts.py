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


def expand_with_policy(node, policy_probs):
    # policy_probs: dict move_str->prob
    if node.state.is_terminal():
        return node, 1.0 if node.state.result() == 1 else -1.0 if node.state.result() == -1 else 0.0

    # policy_probs: dict mapping move_str to prior probability
    from Jeu.Yolah import Move  # local import to avoid cycle at top
    for move_str, prob in policy_probs.items():
        # create child state by cloning and playing the move
        s2 = node.state.clone()
        m = Move.from_str(move_str)
        s2.play(m)
        node.children[move_str] = AlphaNode(s2, parent=node, move=m, prior=prob)

    return node, None


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

        # collect a batch of leaves for batched evaluation
        leaves = [select(root, c_puct=c_puct)]
        # try to gather more leaves by repeating selection (non-destructive)
        for _ in range(15):
            l = select(root, c_puct=c_puct)
            if l is leaves[0]:
                break
            leaves.append(l)

        # filter out terminal leaves and already-expanded ones
        eval_leaves = [l for l in leaves if (not l.state.is_terminal()) and (not l.children)]

        if not eval_leaves:
            # fallback: handle a single selected leaf
            leaf = leaves[0]
            if leaf.state.is_terminal():
                result = leaf.state.result()
                value = result if leaf.state.current_player() == 0 else -result
                backpropagate(leaf, value)
                it += 1
                continue

            # single evaluation (no batch available or no eligible leaves)
            try:
                if hasattr(evaluator, 'batch_eval'):
                    policy_probs, value = evaluator.batch_eval([leaf])[0]
                else:
                    policy_probs, value = evaluator(leaf.state)
            except Exception:
                policy_probs, value = evaluator(leaf.state)

            node_after_expand, _ = expand_with_policy(leaf, policy_probs)
            if node_after_expand.children:
                child = max(node_after_expand.children.values(), key=lambda ch: ch.prior)
                if value is None:
                    try:
                        _, value = evaluator(leaf.state)
                    except Exception:
                        value = 0.0
                backpropagate(child, value)
            else:
                if value is None:
                    try:
                        _, value = evaluator(leaf.state)
                    except Exception:
                        value = 0.0
                backpropagate(leaf, value)

            it += 1
            continue

        # evaluate batch either via evaluator.batch_eval if available, else fall back to single evals
        results = []
        if hasattr(evaluator, 'batch_eval'):
            try:
                results = evaluator.batch_eval(eval_leaves)
            except Exception:
                results = [evaluator(l.state) for l in eval_leaves]
        else:
            results = [evaluator(l.state) for l in eval_leaves]

        # expand each leaf with corresponding policy and backpropagate its value
        for l, (policy_probs, value) in zip(eval_leaves, results):
            node_after_expand, _ = expand_with_policy(l, policy_probs)
            if node_after_expand.children:
                child = max(node_after_expand.children.values(), key=lambda ch: ch.prior)
                # if evaluator.batch_eval doesn't provide a value (ov fallback), we may have value None
                if value is None:
                    # try to get scalar value from a single evaluator call
                    try:
                        _, value = evaluator(l.state)
                    except Exception:
                        value = 0.0
                backpropagate(child, value)
            else:
                # no children (shouldn't happen), use value at leaf
                if value is None:
                    try:
                        _, value = evaluator(l.state)
                    except Exception:
                        value = 0.0
                backpropagate(l, value)

        it += len(eval_leaves)

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
