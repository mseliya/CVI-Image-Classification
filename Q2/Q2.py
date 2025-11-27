import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Load MNIST Data
# -------------------------

train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# First column = label, rest = pixels
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values / 255.0  # normalize

y_test = test_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values / 255.0

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# -------------------------
# 1. Logistic Regression
# -------------------------
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="multinomial")
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_acc)

# -------------------------
# 2. Neural Network (MLP)
# -------------------------
print("\nTraining MLP Neural Network...")
mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128),
                          activation='relu',
                          solver='adam',
                          max_iter=15)

mlp_model.fit(X_train, y_train)

mlp_pred = mlp_model.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
print("Neural Network Accuracy:", mlp_acc)

# -------------------------
# Final Summary
# -------------------------
print("\n------ FINAL ACCURACIES ------")
print("Logistic Regression:", lr_acc)
print("Neural Network (MLP):", mlp_acc)
