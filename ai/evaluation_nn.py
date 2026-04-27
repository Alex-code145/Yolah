import numpy as np
import tensorflow as tf

from ai.features import extract_features
from ai.evaluation import evaluate as fallback_evaluate

MODEL_PATH = "data/yolah_value_model.keras"
_model = None

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def evaluate_nn(state, player):
    try:
        model = load_model()
        x = np.array([extract_features(state, player)], dtype=np.float32)

        # Plus rapide que model.predict pour une seule position
        probs = model(x, training=False).numpy()[0]

        return float(probs[2] - probs[0])
    except Exception as e:
        print("NN evaluation error:", e)
        return fallback_evaluate(state, player)