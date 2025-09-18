# Task 1: Setting Up Machine Learning Environment

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

a = pd.DataFrame(x, columns=iris.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.3
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)