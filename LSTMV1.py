import pandas as pd
import tensorflow as tf
import numpy as np

# Hyper parameters
batch_size = 55

# Data - needs to be in shape of 3d tensor made up of slices of 2d array
btc_data = pd.read_csv('coindata/BTCData.csv')
btc_data = btc_data.select_dtypes(include='float')
state_size = len(btc_data.columns)
x_test = btc_data[:300]
x_train = btc_data[300:]

# Convert to tensors
x_test = tf.constant(x_test.to_numpy(), tf.float32)
x_train = tf.constant(x_train.to_numpy(), tf.float32)

# Labels need to represent output data
# Output will be candle trend data 1 for positive and 0 for negative

labels = []
for index, row in btc_data.iterrows():
    if row['Open'] < row['Close']:
        labels.append(1)
    else:
        labels.append(0)

# Split labels for testing and training
y_test = labels[:300]
y_train = labels[300:]

# Convert to tensors
y_test = tf.constant(y_test, tf.int8)
y_train = tf.constant(y_train, tf.int8)

# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(770, input_shape=[batch_size, state_size]))
model.add(tf.keras.layers.Dense(21, activation='softmax'))
model.build()
model.summary()


# Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# # Fit model
# # model.fit(x_train, y_train,
#           validation_set=(x_test, y_test),
#           batch_size=batch_size,
#           epochs=5)

