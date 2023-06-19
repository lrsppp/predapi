import numpy as np
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class SimpleCNN:
    def __init__(self, num_filters, filter_size, pool_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
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

    def train(self, train_data, train_labels, epochs):
        self.model.fit(train_data, train_labels, epochs=epochs)

    def save_model(self, file_path):
        self.model.save(file_path)


# from utils import load_data


def main():
    num_filters = 8
    filter_size = 2
    pool_size = 2

    # Create an instance of SimpleCNN
    cnn_object = SimpleCNN(num_filters, filter_size, pool_size)

    # Load your training data and labels
    # data = load_data("data.npy")
    # train_data = ...
    # train_labels = ...
    # test_data = ...
    # test_labels = ...
    # X = np.asarray([x.reshape(20, 20) for x in X], dtype=np.float32)
    # y = np.asarray(y, dtype=np.int32)

    # Train the model
    epochs = 10
    cnn_object.train(train_data, train_labels, epochs)

    # Save the model
    model_file_path = "path/to/save/model.h5"
    cnn_object.save_model(model_file_path)


if __name__ == "__main__":
    main()
