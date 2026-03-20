# cython: language_level=3
from libc.stdlib cimport rand, srand
from libc.time cimport time as ctime
cdef unsigned long long FILE_A = 0x0101010101010101
cdef unsigned long long FILE_H = FILE_A << 7

cdef inline unsigned long long bit_not(unsigned long long n):
    return (~n) & 0xFFFFFFFFFFFFFFFF

cdef inline unsigned long long shift_north(unsigned long long b):
    return (b << 8) & 0xFFFFFFFFFFFFFFFF

cdef inline unsigned long long shift_south(unsigned long long b):
    return b >> 8

cdef inline unsigned long long shift_east(unsigned long long b):
    return ((b & bit_not(FILE_H)) << 1) & 0xFFFFFFFFFFFFFFFF

cdef inline unsigned long long shift_west(unsigned long long b):
    return (b & bit_not(FILE_A)) >> 1

cdef inline unsigned long long shift_ne(unsigned long long b):
    return ((b & bit_not(FILE_H)) << 9) & 0xFFFFFFFFFFFFFFFF

cdef inline unsigned long long shift_se(unsigned long long b):
    return (b & bit_not(FILE_H)) >> 7

cdef inline unsigned long long shift_sw(unsigned long long b):
    return (b & bit_not(FILE_A)) >> 9

cdef inline unsigned long long shift_nw(unsigned long long b):
    return ((b & bit_not(FILE_A)) << 7) & 0xFFFFFFFFFFFFFFFF


cdef int is_terminal_c(unsigned long long black, unsigned long long white, unsigned long long empty):
    cdef unsigned long long possible = bit_not(black) & bit_not(white) & bit_not(empty)
    if (shift_north(black) & possible) != 0: return 0
    if (shift_east(black) & possible) != 0: return 0
    if (shift_south(black) & possible) != 0: return 0
    if (shift_west(black) & possible) != 0: return 0
    if (shift_ne(black) & possible) != 0: return 0
    if (shift_se(black) & possible) != 0: return 0
    if (shift_sw(black) & possible) != 0: return 0
    if (shift_nw(black) & possible) != 0: return 0
    if (shift_north(white) & possible) != 0: return 0
    if (shift_east(white) & possible) != 0: return 0
    if (shift_south(white) & possible) != 0: return 0
    if (shift_west(white) & possible) != 0: return 0
    if (shift_ne(white) & possible) != 0: return 0
    if (shift_se(white) & possible) != 0: return 0
    if (shift_sw(white) & possible) != 0: return 0
    if (shift_nw(white) & possible) != 0: return 0
    return 1


def optimized_rollout(state_tuple):
    """Perform a randomized rollout starting from the given state tuple.
    state_tuple should be (black, white, empty, black_score, white_score, ply)
    Returns 1 if black wins, -1 if white wins, 0 if tie.
    """
    cdef unsigned long long black = <unsigned long long> state_tuple[0]
    cdef unsigned long long white = <unsigned long long> state_tuple[1]
    cdef unsigned long long empty = <unsigned long long> state_tuple[2]
    cdef int black_score = <int> state_tuple[3]
    cdef int white_score = <int> state_tuple[4]
    cdef int ply = <int> state_tuple[5]

    # seed rand with time and address-derived value
    srand(<unsigned int> (ctime(NULL) ^ (<unsigned long> &black)))

    while not is_terminal_c(black, white, empty):
        free = bit_not(black | white | empty)
        current_black_turn = 1 if (ply & 1) == 0 else 0

        # gather moves as Python list of tuples (from_bb, to_bb)
        moves = []
        bitboard = black if current_black_turn else white
        b = bitboard
        while b:
            pos = b & -b
            # explore directions
            dst = shift_north(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_north(dst)
            dst = shift_east(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_east(dst)
            dst = shift_south(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_south(dst)
            dst = shift_west(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_west(dst)
            dst = shift_ne(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_ne(dst)
            dst = shift_se(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_se(dst)
            dst = shift_sw(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_sw(dst)
            dst = shift_nw(pos)
            while dst & free:
                moves.append((pos, dst))
                dst = shift_nw(dst)
            b ^= pos

        if not moves:
            # pass move
            ply += 1
            continue

        idx = rand() % len(moves)
        from_bb, to_bb = moves[idx]

        if current_black_turn:
            black = (black & bit_not(from_bb)) | to_bb
            black_score += 1
        else:
            white = (white & bit_not(from_bb)) | to_bb
            white_score += 1
        empty |= from_bb
        ply += 1

    # result
    if black_score > white_score:
        return 1
    elif white_score > black_score:
        return -1
    else:
        return 0
