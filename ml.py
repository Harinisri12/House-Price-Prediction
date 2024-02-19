import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)

columns = ['feature1', 'feature2', 'feature3', 'price']
data = pd.DataFrame(np.column_stack([X, y]), columns=columns)

features = data[['feature1', 'feature2', 'feature3']]
target = data['price']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
print(f'Linear Regression RMSE: {lr_rmse}')

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
nn_predictions = model.predict(X_test)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
print(f'Neural Network RMSE: {nn_rmse}')
