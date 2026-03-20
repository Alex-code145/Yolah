from Yolah import Yolah, Move

def mobility(game, player):
    moves = game.moves_for(player)
    if len(moves) == 1 and moves[0] == Move.none():
        return 0
    return len(moves)

def evaluate(state, player):
    game = state.game
    opponent = Yolah.WHITE_PLAYER if player == Yolah.BLACK_PLAYER else Yolah.BLACK_PLAYER

    if player == Yolah.BLACK_PLAYER:
        my_score = game.black_score
        opp_score = game.white_score
    else:
        my_score = game.white_score
        opp_score = game.black_score

    my_mobility = mobility(game, player)
    opp_mobility = mobility(game, opponent)

    score_diff = my_score - opp_score
    mobility_diff = my_mobility - opp_mobility

    return 10 * score_diff + mobility_diff
