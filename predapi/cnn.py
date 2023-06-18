from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical


num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
    Conv2D(num_filters, filter_size, padding='same', activation='relu', input_shape=(20, 20, 1)),
    MaxPooling2D(pool_size=2),
    Flatten()
    Dense(10, activation="softmax"),
])



model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)