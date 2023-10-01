#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate the data
X = np.random.rand(100, 1)
Y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Create a DataFrame
df = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten()})
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[['X']]
y_train = train_df['Y']
X_test = test_df[['X']]
y_test = test_df['Y']

pipeline = PMMLPipeline([
    ("regressor", LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

sklearn2pmml(pipeline, "lr_model.pmml", with_repr = True)