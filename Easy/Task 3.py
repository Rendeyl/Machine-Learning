# Task 3: Implementing Linear Regression

# Goals:
# Split data into training and testing sets
# Train a linear regression model using scikit-learn
# Evaluate with RMSE and R² metrics

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load the Dataset
data = load_diabetes()
x, y = data.data, data.target

# Split & Train 80% & Test 20%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# R² shows how well the model explains the data (closer to 1 = better)
# RMSE shows the average prediction error (lower = better)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("RMSE:", rmse)
