# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:08:28 2024

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

df = pd.read_csv('Bilayers_Feat_C2DB_CONTCARS.csv', encoding='latin-1')
df.info()

# Split X and y
X = df.iloc[:, 1:-2]
y_low = df.iloc[:, -2]
y_high = df.iloc[:, -1]

# find the NaN
non_nan_indices = y_high.dropna().index
X_high = X.loc[non_nan_indices]
y_high = y_high.loc[non_nan_indices]

# Split the test set
test_indices = np.random.choice(non_nan_indices, size=100, replace=False)
X_test = X.loc[test_indices]
y_test = y_low.loc[test_indices]
yh_test = y_high.loc[test_indices]

# Remove the test set from the rest of the data
train_indices = non_nan_indices.difference(test_indices)
Xh_train = X.loc[train_indices]
yh_train = y_high.loc[train_indices]

# Make the rest of the data the training set
y_train = y_low.drop(test_indices)
X_train = X.drop(test_indices)

# Convert to float
X_train = X_train.values.astype(float)
X_test = X_test.values.astype(float)
y_train = y_train.values.astype(float)
y_test = y_test.values.astype(float)
Xh_train = Xh_train.values.astype(float)
yh_train = yh_train.values.astype(float)
yh_test = yh_test.values.astype(float)

# Scaling the variables
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Define the model building function
def build_model(hp):
    model = Sequential()
    
    units = hp.Int('units', min_value=0, max_value=500, step=10)
    #dropout_rate = hp.Float('dropout', min_value=0.4, max_value=0.5, step=0.1)
    
    model.add(Dense(units=units, activation='relu', input_dim=22))
    #model.add(Dropout(rate=dropout_rate))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return model

tuner = kt.RandomSearch(build_model,
                        objective='val_mean_squared_error',
                        max_trials=10,
                        executions_per_trial=2,
                        directory='Step2',
                        project_name='Units_tuning')

tuner.search(Xh_train, yh_train, epochs=100, validation_split=0.2)

best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

best_model.fit(Xh_train, yh_train, batch_size=10, epochs=100)

#Evaluate model
test_loss, test_mse = best_model.evaluate(Xh_train, yh_train)
y_pred = best_model.predict(X_test)

# Calculate R² for train and test set
train_r2 = r2_score(yh_train, best_model.predict(Xh_train))
test_r2 = r2_score(yh_test, y_pred)

print("Train R² Score:", train_r2)
print("Test R² Score:", test_r2)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.plot([min(yh_test), max(yh_test)], [min(yh_test), max(yh_test)], color='red', linestyle='--')

plt.scatter(yh_test, y_pred, color='blue')

plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.title('y_test vs y_predict')

plt.grid(True)
plt.show()







