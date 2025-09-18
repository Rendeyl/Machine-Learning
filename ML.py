import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

a = pd.DataFrame(x, columns=iris.feature_names)
print(a)