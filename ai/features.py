from Yolah import Yolah, Move

CENTER_SQUARES = [27, 28, 35, 36]
EXTENDED_CENTER = [18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45]

def mobility(game, player):
    moves = game.moves_for(player)
    if len(moves) == 1 and moves[0] == Move.none():
        return 0
    return len(moves)

def count_on_squares(bitboard, squares):
    return sum(1 for sq in squares if bitboard & (1 << sq))

def extract_features(state, player):
    game = state.game
    opponent = Yolah.WHITE_PLAYER if player == Yolah.BLACK_PLAYER else Yolah.BLACK_PLAYER

    if player == Yolah.BLACK_PLAYER:
        my_score = game.black_score
        opp_score = game.white_score
        my_bb = game.black
        opp_bb = game.white
    else:
        my_score = game.white_score
        opp_score = game.black_score
        my_bb = game.white
        opp_bb = game.black

    my_mobility = mobility(game, player)
    opp_mobility = mobility(game, opponent)

    my_piece_count = my_bb.bit_count()
    opp_piece_count = opp_bb.bit_count()

    my_center = count_on_squares(my_bb, CENTER_SQUARES)
    opp_center = count_on_squares(opp_bb, CENTER_SQUARES)

    my_ext_center = count_on_squares(my_bb, EXTENDED_CENTER)
    opp_ext_center = count_on_squares(opp_bb, EXTENDED_CENTER)

    empty_count = game.empty.bit_count()
    occupied_count = game.black.bit_count() + game.white.bit_count()
    free_count = 64 - occupied_count - empty_count

    my_blocked = 1 if my_mobility == 0 else 0
    opp_blocked = 1 if opp_mobility == 0 else 0

    return [
        my_score - opp_score,
        my_score,
        opp_score,

        my_mobility - opp_mobility,
        my_mobility,
        opp_mobility,

        my_blocked - opp_blocked,
        my_blocked,
        opp_blocked,

        my_piece_count - opp_piece_count,
        my_piece_count,
        opp_piece_count,

        my_center - opp_center,
        my_center,
        opp_center,

        my_ext_center - opp_ext_center,
        my_ext_center,
        opp_ext_center,

        empty_count,
        free_count,
        game.ply / 64.0,
    ]