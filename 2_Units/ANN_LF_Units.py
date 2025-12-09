# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:09:28 2024

@author: brand
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

#from tensorflow.keras import layers
#from kerastuner.tuners import RandomSearch
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from kerastuner.tuners import RandomSearch

# Data
dataset = pd.read_csv('C2DB_PCA.csv')

X = dataset.iloc[:, 1:-1].values  # Exclude the last column
y = dataset.iloc[:, -1].values   # Use only the last column

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the variables
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Define the model building function
def build_model(hp):
    model = Sequential()
    
    units = hp.Int('units', min_value=1, max_value=500, step=10)
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    
    model.add(Dense(units=units, activation='relu', input_dim=350))
    model.add(Dropout(rate=dropout_rate))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return model

tuner = kt.RandomSearch(build_model,
                        objective='val_mean_squared_error',
                        max_trials=10,
                        executions_per_trial=2,
                        directory='Step2',
                        project_name='Units_tuning')

tuner.search(X_train, y_train, epochs=100, validation_split=0.2)

best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

best_model.fit(X_train, y_train, batch_size=10, epochs=100)

#Evaluate model
test_loss, test_mse = best_model.evaluate(X_test, y_test)
y_pred = best_model.predict(X_test)

# Calculate R² for train and test set
train_r2 = r2_score(y_train, best_model.predict(X_train))
test_r2 = r2_score(y_test, y_pred)

print("Train R² Score:", train_r2)
print("Test R² Score:", test_r2)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

plt.scatter(y_test, y_pred, color='blue')

plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.title('y_test vs y_predict')

plt.grid(True)
plt.show()







