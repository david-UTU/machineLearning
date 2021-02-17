# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:03:36 2021

@author: dgary
"""

'WRITTEN AS A LEARNING RESOURCE USING AN ONLINE TUTORIAL'

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

###LSTM Learning###

data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
    #generates 100 vectors of 5 consecutive digits
target = [(i+5)/100 for i in range(100)]
    #generates what would be the 6th digit in the sequence
    #Both of these are divided by 100 to normalize the data
data = np.array(data, dtype = float)
target = np.array(target, dtype = float)

#################################################
#Now we split the data into testing and training#
#################################################
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 4)
    #trains 20% of the dataset
    #setting a random state means your tests are deterministic
model = Sequential()
model.add(LSTM((1), batch_input_shape=(None, 5, 1), return_sequences=True))
    #batch_input_shape composition:
        #the number of inputs in your data (100), you can also put in none to make it versatile
        #length of your input sequence (5)
        #then also the length of each vector (1 each vector here is 1x1)
    #setting return_sequences to false simply returns one output
model.compile(loss = 'mean_absolute_error', optimizer="adam", metrics = ['accuracy'])
history = model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test))

results = model.predict(x_test)
plt.scatter(range(20), results, c = 'r')
    #red results
plt.scatter(range(20), y_test, c = 'b')
    #blue y values
