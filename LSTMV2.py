import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import stuff

# Hyper parameters
BATCH_SIZE = 21
EPOCHS = 5
BUFFER_SIZE = 55

# Data - needs to be in shape of 3d tensor made up of slices of 2d array
btc_data = pd.read_csv('coindata/BTCData.csv')
btc_data = btc_data.select_dtypes(include='float')
STATE_SIZE = len(btc_data.columns)
TOTAL_LENGTH = btc_data.__len__() - BUFFER_SIZE
btc_data_step = np.empty([TOTAL_LENGTH, BUFFER_SIZE, STATE_SIZE])

for i in range(TOTAL_LENGTH):
    btc_data_step[i] = btc_data[i:i + BUFFER_SIZE].to_numpy()




btc_data_step = tf.constant(btc_data_step, tf.float64)
print(tf.shape(btc_data_step))


#
# # Build Model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(770, input_shape=[BUFFER_SIZE, STATE_SIZE]))
# model.add(tf.keras.layers.Dense(256))
# model.add(tf.keras.layers.Dense(55))
# model.build()
#
# # Compile model
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.CategoricalCrossentropy())
#
# # Fit model
# single_step_history = model.fit(x_train, y_train,
#                                 validation_data=(x_test, y_test),
#                                 batch_size=BATCH_SIZE,
#                                 epochs=EPOCHS)
#
#
# def plot_train_history(history, title):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     epochs = range(len(loss))
#
#     plt.figure()
#
#     plt.plot(epochs, loss, 'b', label='Training loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation loss')
#     plt.title(title)
#     plt.legend()
#
#     plt.show()
#
#
# plot_train_history(single_step_history,
#                    'Single Step History and Validation')
