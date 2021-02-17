# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 20:48:56 2021

@author: dgary
"""

'WRITTEN AS A LEARNING RESOURCE USING AN ONLINE TUTORIAL'

import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = pd.read_csv('GME.csv', header=0, usecols=['Date', 'Close'], parse_dates=True, index_col='Date')
print(data)

plt.plot(data['Close'])
plt.show()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
    #This scales our data down and normalizes it, 
    #which allows us to use it for training the model more easily
print(scaled_data)

train_length = int(len(scaled_data) * 0.7)
    #we only run this for 70% of the data
test_length = len(scaled_data) - train_length
    #Tests for 30%. This is where our predictions happen
train_data = scaled_data[0:train_length,:]
    #Specifying the train data
test_data = scaled_data[train_length:len(scaled_data),:]
    #Specifying the testing data

def create_dataset(dataset, timestep=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - timestep - 1):
        #timestep is number of days, since that is how the sheet works
        data_x.append(dataset[i:(i+timestep),0])
            #current day's data(closing price)
        data_y.append(dataset[i+timestep,0])
            #next day's closing price
    return np.array(data_x), np.array(data_y)

timestep = 1
train_x, train_y = create_dataset(train_data, timestep)
test_x, test_y = create_dataset(test_data, timestep)

train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

#############################
###Time to Build the Model###
#############################

model = Sequential()
model.add(LSTM(256, input_shape=(1,1)))
model.add(Dense(1, activation='sigmoid'))
    #sigmoid converts final output to a probability score
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#############################
###Time to train the model###
#############################

model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=1)
score = model.evaluate(train_x, train_y, verbose=0)
print('Keras model loss = ', score[0])
print('Keras model accuracy = ', score[1])

train_predictions = model.predict(train_x)
test_predictions = model.predict(test_x)

print(train_predictions)

#######################
###Unscale our model###
#######################

train_predictions = scaler.inverse_transform(train_predictions)
train_y = scaler.inverse_transform([train_y])

test_predictions = scaler.inverse_transform(test_predictions)
test_y = scaler.inverse_transform([test_y])

#############################
###Lay Out Our Predictions###
#############################

print(train_predictions)
train_predict_plot = np.empty_like(scaled_data)
    #creates an empty array framework
train_predict_plot[:,:] = np.nan
    #ensures there are no unwanted values
train_predict_plot[1:len(train_predictions)+1, :] = train_predictions
    #inserting our train predictions
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:,:] = np.nan
test_predict_plot[len(train_predictions)+2+1:len(scaled_data)-1, :] = test_predictions

plt.plot(scaler.inverse_transform(scaled_data))
    #blue line
plt.plot(train_predict_plot, c = 'r')
    #red line
plt.plot(test_predict_plot, c = 'g')
    #green line
plt.show()