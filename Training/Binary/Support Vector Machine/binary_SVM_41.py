import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.svm import SVC  # Import the Support Vector Classifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)

#-----VARIABLES TO MODIFY-----#
feature_count = 41
ml_algo = "Support Vector Machine"
ml_algo_short = "SVM"

# For hyperparameter tuning: you can tune C, kernel, gamma, etc.
clf = SVC(kernel='linear', random_state=42)  # Example with linear kernel
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

    # Display confusion matrix and normalized version
    conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    conf_matrix_disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {fold + 1}")
    plt.savefig(f"./results_{feature_count}/conf_matrix_{feature_count}_fold_{fold+1}.png", dpi=300, bbox_inches='tight')
    plt.close()

    norm_conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=norm_conf_matrix, display_labels=[0, 1])
    norm_conf_matrix_disp.plot(cmap=plt.cm.Greens)
    plt.title(f"Normalized Confusion Matrix - Fold {fold + 1}")
    plt.savefig(f"./{results_path}/conf_matrix_norm_{feature_count}_fold_{fold + 1}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n---SCORES---")
print(f"Accuracy: \n{accuracy_scores}")
print(f"Average: {np.mean(accuracy_scores)}")

print(f"\nPrecision (Class 1): \n{precision_scores_class_1}")
print(f"Average: {np.mean(precision_scores_class_1)}")
print(f"Precision (Class 0): \n{precision_scores_class_0}")
print(f"Average: {np.mean(precision_scores_class_0)}")

print(f"\nRecall (Class 1): \n{recall_scores_class_1}")
print(f"Average: {np.mean(recall_scores_class_1)}")
print(f"Recall (Class 0): \n{recall_scores_class_0}")
print(f"Average: {np.mean(recall_scores_class_0)}")

print(f"\nF1 Score (Class 1): \n{f1_scores_class_1}")
print(f"Average: {np.mean(f1_scores_class_1)}")
print(f"F1 Score (Class 0): \n{f1_scores_class_0}")
print(f"Average: {np.mean(f1_scores_class_0)}")

# Aggregate confusion matrices
agg_conf_matrix = np.sum(conf_matrices, axis=0)
agg_norm_conf_matrix = np.round(agg_conf_matrix / np.sum(agg_conf_matrix, axis=1, keepdims=True), decimals=5)

print(f"\n---CONFUSION MATRICES---")
for i, cm in enumerate(conf_matrices):
    print(f"Fold {i+1}:")
    print(np.array2string(cm, separator=', ') + "\n")

print("---AGGREGATED CONFUSION MATRIX---")
print(np.array2string(agg_conf_matrix, separator=', '))

print("\n---NORMALIZED CONFUSION MATRICES---")
for i, cm in enumerate(norm_conf_matrices):
    print(f"Fold {i+1}:")
    print(np.array2string(cm, separator=', ') + "\n")

print("---AGGREGATED NORMALIZED CONFUSION MATRIX---")
print(np.array2string(agg_norm_conf_matrix, separator=', '))

# Display aggregated confusion matrix
agg_conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=agg_conf_matrix, display_labels=[0, 1])
agg_conf_matrix_disp.plot(cmap=plt.cm.Blues)
plt.title("Aggregated Confusion Matrix")
plt.savefig(f"./{results_path}/aggregated_conf_matrix_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

agg_norm_conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=agg_norm_conf_matrix, display_labels=[0, 1])
agg_norm_conf_matrix_disp.plot(cmap=plt.cm.Greens)
plt.title("Aggregated Normalized Confusion Matrix")
plt.savefig(f"./{results_path}/aggregated_conf_matrix_norm_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

# Get predicted values using cross-validation
y_proba_class_1 = cross_val_predict(clf, X, y, cv=kf, method="predict_proba")[:, 1]  # Probabilities for class 1
y_proba_class_0 = cross_val_predict(clf, X, y, cv=kf, method="predict_proba")[:, 0]  # Probabilities for class 0

# Compute ROC-AUC score
roc_auc_class_1 = roc_auc_score(y, y_proba_class_1)
fpr_class_1, tpr_class_1, _ = roc_curve(y, y_proba_class_1)
roc_auc_class_0 = roc_auc_score(y, y_proba_class_0)
fpr_class_0, tpr_class_0, _ = roc_curve(y, y_proba_class_0)

# Print ROC-AUC score
print(f"\n---ROC-AUC SCORES---")
print(f"ROC-AUC Score (Class 1): {roc_auc_class_1}")
print(f"ROC-AUC Score (Class 0): {roc_auc_class_0}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_class_1, tpr_class_1, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_class_1})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve (Class 1)')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f"./{results_path}/roc_auc_curve_{feature_count}_class_1.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(fpr_class_0, tpr_class_0, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_class_0})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve (Class 0)')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f"./{results_path}/roc_auc_curve_{feature_count}_class_0.png", dpi=300, bbox_inches='tight')
plt.close()


# Output in a txt file
with open(f'./{results_path}/results_{ml_algo_short}_{feature_count}.txt', 'w') as file:
    file.write(f"\n---SCORES---")
    file.write(f"\nAccuracy: \n{accuracy_scores}")
    file.write(f"\nAverage: {np.mean(accuracy_scores)}\n")

    file.write(f"\nPrecision (Class 1): \n{precision_scores_class_1}")
    file.write(f"\nAverage: {np.mean(precision_scores_class_1)}\n")
    file.write(f"Precision (Class 0): \n{precision_scores_class_0}")
    file.write(f"\nAverage: {np.mean(precision_scores_class_0)}\n")

    file.write(f"\nRecall (Class 1): \n{recall_scores_class_1}")
    file.write(f"\nAverage: {np.mean(recall_scores_class_1)}\n")
    file.write(f"Recall (Class 0): \n{recall_scores_class_0}")
    file.write(f"\nAverage: {np.mean(recall_scores_class_0)}\n")

    file.write(f"\nF1 Score (Class 1): \n{f1_scores_class_1}")
    file.write(f"\nAverage: {np.mean(f1_scores_class_1)}\n")
    file.write(f"F1 Score (Class 0): \n{f1_scores_class_0}")
    file.write(f"\nAverage: {np.mean(f1_scores_class_0)}\n")

    file.write(f"\n---CONFUSION MATRICES---")
    for i, cm in enumerate(conf_matrices):
        file.write(f"\nFold {i+1}:\n")
        file.write(np.array2string(cm, separator=', ') + "\n")

    file.write("\n---AGGREGATED CONFUSION MATRIX---\n")
    file.write(f"{agg_conf_matrix}\n")

    file.write("\n---NORMALIZED CONFUSION MATRICES---")
    for i, cm in enumerate(norm_conf_matrices):
        file.write(f"\nFold {i+1}:\n")
        file.write(np.array2string(cm, separator=', ') + "\n")

    file.write("\n---AGGREGATED NORMALIZED CONFUSION MATRIX---\n")
    file.write(f"{agg_norm_conf_matrix}\n")

    file.write(f"\n---ROC-AUC SCORES---")
    file.write(f"\nROC-AUC Score (Class 1): {roc_auc_class_1}")
    file.write(f"\nROC-AUC Score (Class 0): {roc_auc_class_0}\n")

    file.write(f"\nTime it took to execute (in seconds): {time.time() - start_time:.4f}")


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")
