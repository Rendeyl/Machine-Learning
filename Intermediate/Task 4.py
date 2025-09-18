# Task 4: Classification with Decision Trees

# Goals:
# Train a decision tree model
# Evaluate using accuracy, precision, recall, and confusion matrix
# Visualize the decision tree

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load DataSet
d = sns.load_dataset("titanic")

# Only use the needed data
d = d[["survived", "pclass", "sex", "age", "fare", "alone"]].copy()

# Change categorical data into numerical using .map()
d["sex"] = d["sex"].map({"male": 0, "female": 1})
d["alone"] = d["alone"].map({False: 1, True: 0})

# Fill in the missing age data with median
d["age"] = d["age"].fillna(d["age"].median())

# Assign x & y as features and targets
x = d.drop("survived", axis=1)
y = d["survived"]

# Split & Traint 80% and Test 20%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

tree = DecisionTreeClassifier(random_state=42, max_depth=4)
tree.fit(x_train, y_train)

y_pred = tree.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)

print("")

print("Confusion Matrix:\n", cm)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Died (0)", "Survived (1)"], yticklabels=["Died (0)", "Survived (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Visualize the Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(tree, feature_names=x.columns, class_names=["Died", "Survived"], filled=True, rounded=True)
plt.show()

