import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import stuff

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

# Labels need to represent output data
# Output will be candle trend data 1 for positive and 0 for negative
labels = []
for index, row in btc_data.iterrows():
    if row['Open'] < row['Close']:
        labels.append(1)
    else:
        labels.append(0)

# Split labels for testing and training
labels = labels[:x_test.__len__() + x_train.__len__()]
y_test = labels[:330]
y_train = labels[330:]
Y_TEST_DEPTH = int(len(y_test) / BATCH_SIZE)
Y_TRAIN_DEPTH = int(len(y_train) / BATCH_SIZE)

# Convert to tensors
y_test = tf.constant(y_test, tf.int64)
y_train = tf.Variable(y_train, tf.int64)
x_test = tf.constant(x_test.to_numpy(), tf.float64)
x_train = tf.Variable(x_train.to_numpy(), tf.float64)


# Batch all data
x_test = tf.reshape(x_test, [X_TEST_DEPTH, BATCH_SIZE, STATE_SIZE])
y_test = tf.reshape(y_test, [Y_TEST_DEPTH, BATCH_SIZE, 1])
x_train = tf.reshape(x_train, [X_TRAIN_DEPTH, BATCH_SIZE, STATE_SIZE])
y_train = tf.reshape(y_train, [Y_TRAIN_DEPTH, BATCH_SIZE, 1])

# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(770, input_shape=[BATCH_SIZE, STATE_SIZE]))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(55))
model.build()

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy())

# Fit model
single_step_history = model.fit(x_train, y_train,
                                validation_data=(x_test, y_test),
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS)


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


plot_train_history(single_step_history,
                   'Single Step History and Validation')

