import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)

#-----VARIABLES TO MODIFY-----#
feature_count = 11
ml_algo = "Random Forest"
ml_algo_short = "RF"
k_folds = 10  # Number of folds

results_path = f"results_{feature_count}"
os.makedirs(results_path, exist_ok=True)
#-----------------------------#

# Tracks execution time
start_time = time.time()

print(f"Training {ml_algo} using {feature_count} Features...")

# Load the dataset
df = pd.read_csv(f"../../training_data_{feature_count}.csv")

# Split dataset into features (X) and target variable (y)
X = df.drop(columns=["Traffic", "Target"])
y = df["Target"]

# Set up k-fold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

# Hyperparameter tuning using GridSearchCV
print(f"\nPerforming Hyperparameter Tuning for {ml_algo_short}...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid,
                           cv=kf,
                           scoring="accuracy",
                           n_jobs=-1)
grid_search.fit(X, y)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Parameters for {ml_algo_short}: {best_params}")

# Use the best parameters from Grid Search
clf = RandomForestClassifier(**best_params, random_state=42)

#-----------------------------#

# Lists to store confusion matrices and scores
conf_matrices, norm_conf_matrices = [], []
accuracy_scores = []
precision_scores_class_1, precision_scores_class_0 = [], []
recall_scores_class_1, recall_scores_class_0 = [], []
f1_scores_class_1, f1_scores_class_0 = [], []

# Perform manual K-Fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nProcessing Fold {fold + 1}/{k_folds}...")

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test fold
    y_pred = clf.predict(X_val)

    # Compute confusion matrix for the current fold
    conf_matrix = confusion_matrix(y_val, y_pred)
    conf_matrices.append(conf_matrix)
    norm_conf_matrix = np.round(conf_matrix / np.sum(conf_matrix, axis=1).reshape(-1, 1), decimals=5)
    norm_conf_matrices.append(norm_conf_matrix)

    # Compute scores
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    precision_scores_class_1.append(precision_score(y_val, y_pred, pos_label=1))
    precision_scores_class_0.append(precision_score(y_val, y_pred, pos_label=0))
    recall_scores_class_1.append(recall_score(y_val, y_pred, pos_label=1))
    recall_scores_class_0.append(recall_score(y_val, y_pred, pos_label=0))
    f1_scores_class_1.append(f1_score(y_val, y_pred, pos_label=1))
    f1_scores_class_0.append(f1_score(y_val, y_pred, pos_label=0))

# Display aggregated confusion matrix
agg_conf_matrix = np.sum(conf_matrices, axis=0)
agg_norm_conf_matrix = np.round(agg_conf_matrix / np.sum(agg_conf_matrix, axis=1, keepdims=True), decimals=5)

# ROC-AUC
y_proba = cross_val_predict(clf, X, y, cv=kf, method="predict_proba")
roc_auc_class_1 = roc_auc_score(y, y_proba[:, 1])
roc_auc_class_0 = roc_auc_score(y, y_proba[:, 0])

# Output in a text file
with open(f'./{results_path}/results_{ml_algo_short}_{feature_count}.txt', 'w') as file:
    file.write(f"Best Parameters: {best_params}\n")
    file.write(f"Accuracy: {np.mean(accuracy_scores)}\n")
    file.write(f"Precision (Class 1): {np.mean(precision_scores_class_1)}\n")
    file.write(f"Precision (Class 0): {np.mean(precision_scores_class_0)}\n")
    file.write(f"Recall (Class 1): {np.mean(recall_scores_class_1)}\n")
    file.write(f"Recall (Class 0): {np.mean(recall_scores_class_0)}\n")
    file.write(f"F1 Score (Class 1): {np.mean(f1_scores_class_1)}\n")
    file.write(f"F1 Score (Class 0): {np.mean(f1_scores_class_0)}\n")
    file.write(f"ROC-AUC Score (Class 1): {roc_auc_class_1}\n")
    file.write(f"ROC-AUC Score (Class 0): {roc_auc_class_0}\n")

# Execution time
end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")
