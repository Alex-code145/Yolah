import math
import random
import time
import multiprocessing
import copy
try:
    from Jeu._yolah_core import optimized_rollout as cython_optimized_rollout
except Exception:
    cython_optimized_rollout = None

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
    return child.value() + c * math.sqrt(
        math.log(parent.visit_count + 1) / (child.visit_count + 1e-9)
    )


def rollout(state):
    # Try using the Cython-optimized rollout when available. The optimized
    # rollout expects a raw state tuple: (black, white, empty, black_score, white_score, ply)
    if cython_optimized_rollout is not None:
        # Extract raw tuple if possible
        state_tuple = None
        if hasattr(state, 'game') and hasattr(state.game, 'get_state'):
            state_tuple = state.game.get_state()
        elif hasattr(state, 'get_state'):
            state_tuple = state.get_state()

        if state_tuple is not None:
            try:
                return cython_optimized_rollout(state_tuple)
            except Exception:
                # fall back to Python rollout on any error
                pass

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


def mcts_collect_stats(root_state, iterations=800, time_limit_s=None):
    root = MCTSNode(root_state)
    start = time.time()
    it = 0

    while True:
        if time_limit_s and time.time() - start > time_limit_s:
            break
        if not time_limit_s and it >= iterations:
            break

        leaf = expand(select(root))
        reward = rollout(leaf.state)
        backpropagate(leaf, reward)
        it += 1

    # Retourne visits + value_sum (PAS value)
    stats = {}
    for ch in root.children:
        stats[str(ch.move)] = (ch.visit_count, ch.value_sum)

    print(f"MCTS: {it} iterations, time taken: {time.time() - start:.2f}s")
    return stats


def _mcts_worker(args):
    """Top-level worker for multiprocessing. Receives (root_state, iterations).
    Returns stats dict for root children (same format as mcts_collect_stats).
    """
    root_state, iterations = args
    # Work on a local copy to avoid accidental shared-state mutation
    local_root = copy.deepcopy(root_state)
    return mcts_collect_stats(local_root, iterations=iterations)


def mcts_parallel(root_state, iterations=800, processes=None):
    """Run MCTS iterations in parallel by creating worker processes.

    Simple strategy: divide `iterations` roughly evenly among `processes` workers.
    Each worker runs independent MCTS from the same root state and returns
    per-child (move) statistics (visit_count, value_sum). We aggregate these
    by summing visit counts and value sums per move string.

    Returns: aggregated stats dict {move_str: (visits_sum, value_sum_sum)}
    """
    if processes is None:
        processes = max(1, multiprocessing.cpu_count() - 1)

    # Divide iterations
    base = iterations // processes
    extras = iterations % processes
    tasks = []
    for i in range(processes):
        iters = base + (1 if i < extras else 0)
        if iters > 0:
            tasks.append((root_state, iters))

    if not tasks:
        return {}

    with multiprocessing.Pool(processes=len(tasks)) as pool:
        results = pool.map(_mcts_worker, tasks)

    # Aggregate results
    agg = {}
    for res in results:
        for mv, (vis, val) in res.items():
            if mv in agg:
                agg[mv] = (agg[mv][0] + vis, agg[mv][1] + val)
            else:
                agg[mv] = (vis, val)

    return agg