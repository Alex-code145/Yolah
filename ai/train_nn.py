import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def build_model(input_dim, nb_hidden=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(nb_hidden, activation="relu"),
        tf.keras.layers.Dense(nb_hidden, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(dataset_path="data/yolah_dataset.csv", model_path="data/yolah_value_model.keras"):
    print("Loading dataset...")

    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    # labels -1, 0, 1 -> 0, 1, 2
    y = y + 1

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_dim=X.shape[1])

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(classification_report(y_test, y_pred))

    os.makedirs("data", exist_ok=True)
    model.save(model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()