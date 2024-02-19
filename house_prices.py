import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)

columns = ['feature1', 'feature2', 'feature3', 'price']
data = pd.DataFrame(np.column_stack([X, y]), columns=columns)

print(data.head())

data.to_csv('house_prices.csv', index=False)
