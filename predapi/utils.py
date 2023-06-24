import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_data(file_path, test_size=0.2, random_state=42):
    X, y = np.load(file_path, allow_pickle=True).T

    X = np.asarray([x.reshape(20, 20) for x in X], dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    y = to_categorical(y)

    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_data, test_data, val_labels, test_labels = train_test_split(
        test_data, test_labels, test_size=0.5, random_state=random_state
    )

    return train_data, train_labels, test_data, test_labels, val_data, val_labels
