import tensorflow as tf
import pandas as pd
# import ta if need to calculate technical analysis again

# Step 1 - Import Data
btcData = pd.read_csv('coinData/BTCUSD.csv')
btcData['Date'] = pd.to_datetime(btcData['Date'])
btcData = btcData.set_index('Date')

print(btcData.dtypes)

# calculate technical analysis for neural network
# btcData = ta.add_all_ta_features(btcData, open='Open', high='High', low='Low', close='Close', volume='Volume BTC', fillna=True)
# print(btcData)
# btcData.to_csv('coinData/BTCData.csv')

