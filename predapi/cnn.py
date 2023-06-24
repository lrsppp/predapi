from predapi.models import ModelConfig, TrainConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


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

