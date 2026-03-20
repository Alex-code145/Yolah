from YolahInterface import YolahState
from ai.minmax import choose_minimax_move

def play_game():
    state = YolahState()

    while not state.is_terminal():
        print(state.game)

        move = choose_minimax_move(state, depth=3)
        print("Chosen move:", move)

        state.play(move)

    print("Final state:")
    print(state.game)
    print("Result:", state.result())

if __name__ == "__main__":
    play_game()