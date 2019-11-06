import pandas as pd
import tensorflow as tf
import numpy as np
#importing stufff

# Hyper parameters
BATCH_SIZE = 55
EPOCHS = 5

# Data - needs to be in shape of 3d tensor made up of slices of 2d array
btc_data = pd.read_csv('coindata/BTCData.csv')
btc_data = btc_data.select_dtypes(include='float')
STATE_SIZE = len(btc_data.columns)
x_test = btc_data[:BATCH_SIZE * 6]
x_train = btc_data[BATCH_SIZE * 6:btc_data.__len__() - (btc_data.__len__() % BATCH_SIZE)]

X_TEST_DEPTH = int(x_test.__len__() / BATCH_SIZE)
X_TRAIN_DEPTH = int(x_train.__len__() / BATCH_SIZE)

# Convert to tensors
x_test = tf.Variable(x_test.to_numpy(), tf.float64)
x_train = tf.Variable(x_train.to_numpy(), tf.float64)

# Labels need to represent output data
# Output will be candle trend data 1 for positive and 0 for negative
labels = []
for index, row in btc_data.iterrows():
    if row['Open'] < row['Close']:
        labels.append(1)
    else:
        labels.append(0)

# Split labels for testing and training
y_test = labels[:330]
y_train = labels[330:]

# Convert to tensors
y_test = tf.constant(y_test, tf.int8)
y_train = tf.constant(y_train, tf.int8)

# Batch all data
# x_test = tf.reshape(x_test, [X_TEST_DEPTH, BATCH_SIZE, STATE_SIZE])
x_train = tf.reshape(x_train, [X_TRAIN_DEPTH, BATCH_SIZE, STATE_SIZE])

# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(770, input_shape=[BATCH_SIZE, STATE_SIZE]))
model.add(tf.keras.layers.Dense(21, activation='softmax'))
model.build()

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adagrad(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy())

# Fit model
model.fit((x_train, y_train), epochs=EPOCHS, validation_data=(x_test, y_test))

