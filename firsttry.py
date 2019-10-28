import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import ta if need to calculate technical analysis again

# hyper parameters
num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length


# Step 1 - Import Data
btcData = pd.read_csv('coinData/BTCUSD.csv')
btcData['Date'] = pd.to_datetime(btcData['Date'])
btcData = btcData.set_index('Date')

print(btcData.dtypes)

# calculate technical analysis for neural network
# btcData = ta.add_all_ta_features(btcData, open='Open', high='High', low='Low', close='Close', volume='Volume BTC', fillna=True)
# print(btcData)
# btcData.to_csv('coinData/BTCData.csv')

# Step 2 - Build Model
batchX_placeholder = tf.placeholder(tf.float32, truncated_backprop_length)
batchY_placeholder = tf.placeholder(tf.int32, truncated_backprop_length)

# input current data and previous states
init_state = tf.placeholder(tf.placeholder(tf.float32, [batch_size, state_size]))

# Weights and Biases
W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

# second set
W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Forward PAss
#state placeholder
current_state = init_state
state_series = []

for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatented = tf.concat(1, [current_input, current_state])
    next_state = tf.tanh(tf.matmul(input_and_state_concatented, W) + b)
    state_series.append(next_state)
    current_state = next_state

# Backward Pass
#calculate loss and minimize it

logits_series = [tf.matmul(state, W2) + b2 for state in state_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epochs_idx in range(num_epochs):
        x, y = btcData

        _current_state = np.zeros((batch_size, state_size))

