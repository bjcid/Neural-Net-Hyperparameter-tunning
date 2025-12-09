# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:09:28 2024

@author: brand
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#from tensorflow.keras import layers
#from kerastuner.tuners import RandomSearch
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from kerastuner.tuners import RandomSearch

# Data
dataset = pd.read_csv('C2DB_PCA.csv')

X = dataset.iloc[:, 1:101].values
y = dataset.iloc[:, 101].values

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Define the model building function
def build_model(hp):
    model = Sequential()
    
    model.add(Dense(50, activation='relu', input_dim=100))
    model.add(Dense(1, activation='relu'))
    
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adadelta'])
    
    model.compile(optimizer= optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])
    
    return model

tuner = kt.RandomSearch(build_model,
                        objective= 'val_mean_absolute_error',
                        max_trials=5)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

tuner.get_best_hyperparameters()[0].values

model = tuner.get_best_models(num_models=1)[0]
model.summary()

model.fit(X_train, y_train, batch_size=10, epochs=100, initial_epoch=50, validation_data=(X_test, y_test))



