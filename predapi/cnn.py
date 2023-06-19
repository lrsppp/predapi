import numpy as np
from tensorflow.keras.utils import to_categorical
from pydantic import BaseSettings

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def load_data(file_path, test_size=0.2, random_state=42):
    X, y = np.load(file_path, allow_pickle=True).T

    X = np.asarray([x.reshape(20, 20) for x in X], dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    y = to_categorical(y)

    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_data, test_data, val_labels, test_labels = train_test_split(
        test_data, test_labels, test_size=0.5, random_state=42
    )

    return train_data, train_labels, test_data, test_labels, val_data, val_labels


class ModelConfig(BaseSettings):
    num_filters: int
    filter_size: int
    pool_size: int


class TrainConfig(BaseSettings):
    epochs: int


class SimpleCNN:
    def __init__(self, config: ModelConfig):
        self.num_filters = config.num_filters
        self.filter_size = config.filter_size
        self.pool_size = config.pool_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential(
            [
                Conv2D(
                    self.num_filters,
                    self.filter_size,
                    padding="same",
                    activation="relu",
                    input_shape=(20, 20, 1),
                ),
                MaxPooling2D(pool_size=self.pool_size),
                Flatten(),
                Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            "adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self, train_data, train_labels, val_data, val_labels, config: TrainConfig
    ):
        epochs = config.epochs
        self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            validation_data=(val_data, val_labels),
        )

    def save_model(self, file_path):
        self.model.save(file_path)


def main():
    config = ModelConfig(num_filters=8, filter_size=2, pool_size=2)
    train_config = TrainConfig(
        epochs=50,
    )
    cnn_object = SimpleCNN(config=config)

    train_data, train_labels, test_data, test_labels, val_data, val_labels = load_data(
        "data.npy"
    )

    cnn_object.train(
        train_data, train_labels, val_data, val_labels, config=train_config
    )

    accuracy = cnn_object.model.evaluate(test_data, test_labels)[1]
    print("Test accuracy:", accuracy)

    # write to file
    model_file_path = "model.h5"
    cnn_object.save_model(model_file_path)


if __name__ == "__main__":
    main()
