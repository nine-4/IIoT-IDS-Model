import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.preprocessing import label_binarize

# -----VARIABLES TO MODIFY----- #
feature_count = 11
ml_algo = "Gaussian Naive Bayes (Multiclass)"
ml_algo_short = "GNB_MC"

k_folds = 10  # Number of folds

results_path = f"results_{feature_count}_multiclass"
os.makedirs(results_path, exist_ok=True)
# ----------------------------- #

# Track execution time
start_time = time.time()

print(f"Training {ml_algo} using {feature_count} Features...")

# Load the dataset
df = pd.read_csv(f"../training_data_{feature_count}.csv")

# Split dataset into features (X) and target variable (y)
# Assuming that "Traffic" is an identifier column and "Target" contains the multiclass labels
X = df.drop(columns=["Traffic", "Target"])
y = df["Target"]

# Get sorted unique class labels for later use in plots
classes = np.sort(y.unique())

# Set up k-fold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

print(f"Performing Hyperparameter Tuning for {ml_algo_short}...")

# Hyperparameter tuning using GridSearchCV
param_grid = {"var_smoothing": np.logspace(-9, 0, 10)}
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=kf, scoring="accuracy", n_jobs=-1)
grid_search.fit(X, y)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Parameters for {ml_algo_short}: {best_params}")

# Use the best parameters found by GridSearchCV
clf = GaussianNB(**best_params)

# Lists to store confusion matrices and scores
conf_matrices = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform manual K-Fold cross-validation
fold_idx = 1
for train_idx, val_idx in kf.split(X):
    print(f"\nProcessing Fold {fold_idx}/{k_folds}...")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the validation fold
    y_pred = clf.predict(X_val)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred, labels=classes)
    conf_matrices.append(conf_matrix)

    # Compute scores using weighted average (can change to 'macro' if preferred)
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    precision_scores.append(precision_score(y_val, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_val, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

    # Plot and save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {fold_idx}")
    plt.savefig(f"./{results_path}/conf_matrix_{feature_count}_fold_{fold_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

    fold_idx += 1

# Aggregate confusion matrices
agg_conf_matrix = np.sum(conf_matrices, axis=0)
agg_norm_conf_matrix = np.round(agg_conf_matrix / np.sum(agg_conf_matrix, axis=1, keepdims=True), decimals=5)

print("\n---AGGREGATED METRICS---")
print(f"Accuracy Scores: {accuracy_scores}")
print(f"Average Accuracy: {np.mean(accuracy_scores)}")

print(f"\nPrecision Scores (Weighted): {precision_scores}")
print(f"Average Precision: {np.mean(precision_scores)}")

print(f"\nRecall Scores (Weighted): {recall_scores}")
print(f"Average Recall: {np.mean(recall_scores)}")

print(f"\nF1 Scores (Weighted): {f1_scores}")
print(f"Average F1 Score: {np.mean(f1_scores)}")

# Display aggregated confusion matrix
disp_agg = ConfusionMatrixDisplay(confusion_matrix=agg_conf_matrix, display_labels=classes)
disp_agg.plot(cmap=plt.cm.Blues)
plt.title("Aggregated Confusion Matrix")
plt.savefig(f"./{results_path}/aggregated_conf_matrix_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

disp_norm_agg = ConfusionMatrixDisplay(confusion_matrix=agg_norm_conf_matrix, display_labels=classes)
disp_norm_agg.plot(cmap=plt.cm.Greens)
plt.title("Aggregated Normalized Confusion Matrix")
plt.savefig(f"./{results_path}/aggregated_conf_matrix_norm_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

# Get predicted probabilities using cross-validation
y_proba = cross_val_predict(clf, X, y, cv=kf, method="predict_proba")

# Binarize the output for ROC computation (one-vs-rest)
y_binarized = label_binarize(y, classes=classes)
n_classes = y_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_proba[:, i])
    roc_auc[i] = roc_auc_score(y_binarized[:, i], y_proba[:, i])
    print(f"ROC-AUC Score for Class {classes[i]}: {roc_auc[i]}")

# Plot all ROC curves in one figure
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve for Multiclass Classification')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f"./{results_path}/roc_auc_curve_multiclass_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

# Output results to a text file
with open(f'./{results_path}/results_{ml_algo_short}_{feature_count}.txt', 'w') as file:
    file.write(f"Best Parameters for {ml_algo_short}: {best_params}\n")
    
    file.write(f"\n---SCORES---")
    file.write(f"\nAccuracy Scores: {accuracy_scores}")
    file.write(f"\nAverage Accuracy: {np.mean(accuracy_scores)}\n")
    
    file.write(f"\nPrecision Scores (Weighted): {precision_scores}")
    file.write(f"\nAverage Precision: {np.mean(precision_scores)}\n")
    
    file.write(f"\nRecall Scores (Weighted): {recall_scores}")
    file.write(f"\nAverage Recall: {np.mean(recall_scores)}\n")
    
    file.write(f"\nF1 Scores (Weighted): {f1_scores}")
    file.write(f"\nAverage F1 Score: {np.mean(f1_scores)}\n")
    
    file.write("\n---AGGREGATED CONFUSION MATRIX---\n")
    file.write(np.array2string(agg_conf_matrix, separator=', ') + "\n")
    
    file.write("\n---AGGREGATED NORMALIZED CONFUSION MATRIX---\n")
    file.write(np.array2string(agg_norm_conf_matrix, separator=', ') + "\n")
    
    file.write("\n---ROC-AUC SCORES PER CLASS---\n")
    for i in range(n_classes):
        file.write(f"\nClass {classes[i]}: {roc_auc[i]}")
    
    file.write(f"\n\nTime it took to execute (in seconds): {time.time() - start_time:.4f}")

end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")
