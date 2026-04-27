from ai.generate_data import generate_dataset
from ai.train_nn import train_model

if __name__ == "__main__":
    X, y = generate_dataset(num_games=1000)
    train_model(X, y)