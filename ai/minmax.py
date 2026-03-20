from ai.evaluation import evaluate

def minimax_alpha_beta(state, depth, alpha, beta, maximizing, root_player):
    if depth == 0 or state.is_terminal():
        return evaluate(state, root_player), None

    legal_moves = state.legal_moves()

    if maximizing:
        best_value = float('-inf')
        best_move = None

        for move in legal_moves:
            child = state.clone()
            child.play(move)

            value, _ = minimax_alpha_beta(
                child,
                depth - 1,
                alpha,
                beta,
                False,
                root_player
            )

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_value, best_move

    else:
        best_value = float('inf')
        best_move = None

        for move in legal_moves:
            child = state.clone()
            child.play(move)

            value, _ = minimax_alpha_beta(
                child,
                depth - 1,
                alpha,
                beta,
                True,
                root_player
            )

            if value < best_value:
                best_value = value
                best_move = move

            beta = min(beta, best_value)
            if beta <= alpha:
                break

        return best_value, best_move


def choose_minimax_move(state, depth=3):
    player = state.current_player()
    _, move = minimax_alpha_beta(
        state,
        depth,
        float('-inf'),
        float('inf'),
        True,
        player
    )
    return move