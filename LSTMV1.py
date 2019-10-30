import tensorflow as tf
import numpy as np

# Hyper parameters

# Data


# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
model.add(tf.keras.layers.LSTM(400))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.build()
model.summary()


# Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Fit model
model.fit(x_train, y_train,
          validation_set=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)