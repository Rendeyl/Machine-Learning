# Task 9: Implementing Principal Component Analysis (PCA)

# Goals:
# Load a high-dimensional dataset
# Apply PCA to reduce features
# Train a classifier before and after PCA
# Compare performance metrics and visualize results

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Original shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier BEFORE PCA
clf_before = LogisticRegression(max_iter=500)
clf_before.fit(X_train_scaled, y_train)
y_pred_before = clf_before.predict(X_test_scaled)

# Apply PCA (reduce to 2 principal components for visualization)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train classifier AFTER PCA
clf_after = LogisticRegression(max_iter=500)
clf_after.fit(X_train_pca, y_train)
y_pred_after = clf_after.predict(X_test_pca)

# Evaluate performance
def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

evaluate_model("Before PCA", y_test, y_pred_before)
evaluate_model("After PCA", y_test, y_pred_after)

# Visualize PCA results
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
plt.title("Data Distribution after PCA (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Explained variance ratio
plt.figure(figsize=(6, 4))
plt.bar(range(1, 3), pca.explained_variance_ratio_, color='skyblue')
plt.title("Explained Variance Ratio by Each Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.show()
