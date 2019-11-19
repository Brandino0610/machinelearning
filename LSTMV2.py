from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import stuff

# Hyper parameters
BATCH_SIZE = 150
EPOCHS = 10
BUFFER_SIZE = 55
TEST_SET_SIZE = 400

# Data - needs to be in shape of 3d tensor made up of slices of 2d array
btc_data = pd.read_csv('coindata/BTCData.csv')
btc_data = btc_data.select_dtypes(include='float')
STATE_SIZE = len(btc_data.columns)
TOTAL_LENGTH = btc_data.__len__() - BUFFER_SIZE

label_data = []
for index, row in btc_data.iterrows():
    if row['Open'] < row['Close']:
        label_data.append(1)
    else:
        label_data.append(0)


label_data_step = np.empty([TOTAL_LENGTH, BUFFER_SIZE])
btc_data_step = np.empty([TOTAL_LENGTH, BUFFER_SIZE, STATE_SIZE])
for i in range(TOTAL_LENGTH):
    btc_data_step[i] = btc_data[i:i + BUFFER_SIZE].to_numpy()
    label_data_step[i] = label_data[i:i + BUFFER_SIZE]


x_train = btc_data_step[:len(btc_data_step) - TEST_SET_SIZE, :, :]
x_test = btc_data_step[len(btc_data_step) - TEST_SET_SIZE:, :, :]
y_train = label_data_step[:len(label_data_step) - TEST_SET_SIZE]
y_test = label_data_step[len(label_data_step) - TEST_SET_SIZE:]

x_train = tf.constant(x_train, tf.float64)
x_test = tf.constant(x_test, tf.float64)
y_train = tf.constant(y_train, tf.float64)
y_test = tf.constant(y_test, tf.float64)


# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(755,
                               input_shape=[BUFFER_SIZE, STATE_SIZE],
                               return_sequences=True))
model.add(tf.keras.layers.LSTM(512))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dense(BUFFER_SIZE))
model.build()

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Fit model
single_step_history = model.fit(x_train, y_train,
                                validation_data=(x_test, y_test),
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,)

# prediction_values = tf.constant(btc_data_step[765], tf.float64)
# prediction_values = tf.reshape(prediction_values, [1, BUFFER_SIZE, STATE_SIZE])
#
# model.predict(tf.constant(prediction_values))


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


plot_train_history(single_step_history, 'Single Step History and Validation')
