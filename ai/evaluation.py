from Yolah import Yolah, Move
from ai.features import extract_features

def evaluate(state, player):
    features = extract_features(state, player)

    score_diff = features[0]
    mobility_diff = features[1]
    piece_diff = features[2]
    center_diff = features[3]

    return (
        10 * score_diff
        + 2 * mobility_diff
        + 1 * piece_diff
        + 3 * center_diff
    )