from Yolah import Yolah


class YolahState:
    def __init__(self, game=None):
        if game is None:
            self.game = Yolah()
        else:
            self.game = game

    def clone(self):
        new_game = Yolah()
        new_game.black, new_game.white, new_game.empty, new_game.black_score, new_game.white_score, new_game.ply = self.game.get_state()
        return YolahState(new_game)

    def legal_moves(self):
        return self.game.moves()

    def play(self, move):
        self.game.play(move)

    def is_terminal(self):
        return self.game.game_over()

    def current_player(self):
        return self.game.current_player()

    def result(self):
        if not self.is_terminal():
            return None
        if self.game.black_score > self.game.white_score:
            return 1
        elif self.game.white_score > self.game.black_score:
            return -1
        else:
            return 0