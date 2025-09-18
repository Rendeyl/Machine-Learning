# Task 1: Setting Up Machine Learning Environment

# Goals:
# Implement a simple classification using k-Nearest Neighbors
# Train the model and evaluate accuracy

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()
x = iris.data
y = iris.target

# Put the features in a DataFrame for viewing
a = pd.DataFrame(x, columns=iris.feature_names)

# Split and Train 70% & Test 30%
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.3
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

# Prediction
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)