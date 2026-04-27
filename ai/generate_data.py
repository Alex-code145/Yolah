import random
import csv
import os

from Yolah import Yolah
from YolahInterface import YolahState
from ai.features import extract_features
from ai.minmax import choose_minimax_move


def random_move(state):
    return random.choice(state.legal_moves())


def mixed_move(state, depth=1, random_ratio=0.3):
    r = random.random()
    ply = state.game.ply

    if ply < 10:
        if r < 0.6:
            return random.choice(state.legal_moves())
        return choose_minimax_move(state, depth=1)

    elif ply < 30:
        if r < random_ratio:
            return random.choice(state.legal_moves())
        elif r < 0.85:
            return choose_minimax_move(state, depth=1)
        return choose_minimax_move(state, depth=2)

    else:
        if r < 0.15:
            return random.choice(state.legal_moves())
        return choose_minimax_move(state, depth=2)


def final_label_for_player(final_state, player):
    result = final_state.result()

    if result == 0:
        return 0

    if player == Yolah.BLACK_PLAYER:
        return 1 if result == 1 else -1
    else:
        return 1 if result == -1 else -1


def generate_dataset(num_games=2000, depth=1, random_ratio=0.5):
    X = []
    y = []

    for game_idx in range(num_games):
        state = YolahState()
        history = []

        while not state.is_terminal():
            player = state.current_player()
            features = extract_features(state, player)
            history.append((features, player))

            move = mixed_move(state, depth=depth, random_ratio=random_ratio)
            state.play(move)

        for features, player in history:
            label = final_label_for_player(state, player)
            X.append(features)
            y.append(label)

        if (game_idx + 1) % 50 == 0:
            print(f"{game_idx + 1} games generated")

    return X, y


def save_dataset(X, y, path="data/yolah_dataset.csv"):
    os.makedirs("data", exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "score_diff", "my_score", "opp_score",
            "mobility_diff", "my_mobility", "opp_mobility",
            "blocked_diff", "my_blocked", "opp_blocked",
            "piece_diff", "my_piece_count", "opp_piece_count",
            "center_diff", "my_center", "opp_center",
            "extended_center_diff", "my_ext_center", "opp_ext_center",
            "empty_count", "free_count", "ply_normalized",
            "label"
        ])

        for features, label in zip(X, y):
            writer.writerow(features + [label])

    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    X, y = generate_dataset(
        num_games=5000,
        depth=1,
        random_ratio=0.3
    )
    save_dataset(X, y)