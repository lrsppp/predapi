from predapi.utils import load_data
from predapi.cnn import SimpleCNN, ModelConfig, TrainConfig
from sklearn.model_selection import train_test_split

TRAIN_CONFIG = TrainConfig(epochs=50)
MODEL_CONFIG = ModelConfig(num_filters=8, filter_size=2, pool_size=2)


def main(data_path, model_config, train_config):
    cnn_object = SimpleCNN(config=model_config)

    train_data, train_labels, test_data, test_labels, val_data, val_labels = load_data(
        data_path
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
    main(
        data_path="data/data.npy", model_config=MODEL_CONFIG, train_config=TRAIN_CONFIG
    )
