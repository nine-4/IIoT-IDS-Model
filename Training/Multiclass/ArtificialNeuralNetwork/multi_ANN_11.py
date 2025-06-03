import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from joblib import dump

# -----VARIABLES TO MODIFY----- #
feature_count = 11
ml_algo = "Artificial Neural Network"
ml_algo_short = "ANN"

k_folds = 10  # Number of folds

results_path = f"results_{feature_count}"
os.makedirs(results_path, exist_ok=True)
# ----------------------------- #

# Tracks execution time
start_time = time.time()

print(f"Training {ml_algo} using {feature_count} Features...")

# Load the dataset
df = pd.read_csv(f"../../training_data_{feature_count}.csv")

# Important: remove the "normal" Traffic samples
df = df[df["Target"] != 0]

df = df.drop_duplicates()

print(f"Dataset size after deduplication: {len(df)}")

# Split dataset into features (X) and target variable (y)
X = df.drop(columns=["Traffic", "Target"])
y = df["Traffic"]   # Important: use "Traffic" column for multiclass

# Set up k-fold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# print(f"Performing Hyperparameter Tuning for {ml_algo_short}...")
#
# # Define hyperparameter grid
# param_grid = {
#     "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)],
#     "activation": ["relu", "tanh"],
#     "solver": ["adam", "sgd"],
#     "max_iter": [500]
# }
#
# # Hyperparameter tuning using GridSearchCV
# grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid, cv=kf, scoring="accuracy", n_jobs=-1)
# grid_search.fit(X, y)
#
# # Best hyperparameters
# best_params = grid_search.best_params_
# print(f"Best Parameters for {ml_algo_short}: {best_params}")

# Use the best parameter/s found by GridSearchCV
clf = MLPClassifier(activation='tanh', hidden_layer_sizes= (100, 50), max_iter= 500, solver= 'adam')
# ----------------------------- #

# Lists to store confusion matrices and scores
conf_matrices, norm_conf_matrices = [], []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform manual K-Fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nProcessing Fold {fold}/10...")

    # Raw splits (no upsampling yet)
    X_train_raw, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train_raw, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    # Combine train for upsampling
    train_df = pd.concat([X_train_raw, y_train_raw], axis=1)
    train_df.columns = list(X_train_raw.columns) + ["label"]

    # Perform upsampling per class
    resampled = []
    max_size = train_df["label"].value_counts().max()
    for label in train_df["label"].unique():
        class_df = train_df[train_df["label"] == label]
        upsampled_df = resample(
            class_df, replace=True, n_samples=max_size, random_state=fold
        )
        resampled.append(upsampled_df)

    train_df_balanced = pd.concat(resampled)
    X_train = train_df_balanced.drop(columns=["label"])
    y_train = train_df_balanced["label"]

    # Check overlap
    overlap = pd.merge(X_train, X_val, how="inner")
    print(f"Number of overlapping rows (data-wise): {len(overlap)}")

    # Print class distributions
    print("Class distribution in Training Set:")
    print(y_train.value_counts().sort_index())
    print("Class distribution in Validation Set:")
    print(y_val.value_counts().sort_index())

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
    precision_scores.append(precision_score(y_val, y_pred, average="weighted"))
    recall_scores.append(recall_score(y_val, y_pred, average="weighted"))
    f1_scores.append(f1_score(y_val, y_pred, average="weighted"))

    # Dynamically get the label of each attack
    attack_labels = clf.classes_

    # Display confusion matrix and normalized version
    conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=attack_labels)
    conf_matrix_disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {fold + 1}")
    plt.savefig(f"./results_{feature_count}/conf_matrix_{feature_count}_fold_{fold+1}.png", dpi=300, bbox_inches='tight')
    plt.close()

    norm_conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=norm_conf_matrix, display_labels=attack_labels)
    norm_conf_matrix_disp.plot(cmap=plt.cm.Greens)
    plt.title(f"Normalized Confusion Matrix - Fold {fold + 1}")
    plt.savefig(f"./{results_path}/conf_matrix_norm_{feature_count}_fold_{fold + 1}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n---SCORES---")
print(f"Accuracy: \n{accuracy_scores}")
print(f"Average: {np.mean(accuracy_scores)}")

print(f"\nPrecision: \n{precision_scores}")
print(f"Average: {np.mean(precision_scores)}")

print(f"\nRecall: \n{recall_scores}")
print(f"Average: {np.mean(recall_scores)}")

print(f"\nF1 Score: \n{f1_scores}")
print(f"Average: {np.mean(f1_scores)}")

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
agg_conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=agg_conf_matrix, display_labels=attack_labels)
agg_conf_matrix_disp.plot(cmap=plt.cm.Blues)
plt.title("Aggregated Confusion Matrix")
plt.savefig(f"./{results_path}/aggregated_conf_matrix_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

agg_norm_conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=agg_norm_conf_matrix, display_labels=attack_labels)
agg_norm_conf_matrix_disp.plot(cmap=plt.cm.Greens)
plt.title("Aggregated Normalized Confusion Matrix")
plt.savefig(f"./{results_path}/aggregated_conf_matrix_norm_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

# Binarize the target labels (One-hot encoding)
y_binarized = label_binarize(y, classes=attack_labels)  # Shape: (n_samples, n_classes)

# Compute ROC AUC score for each class
y_proba = cross_val_predict(clf, X, y, cv=kf, method="predict_proba")  # Get probability estimates
roc_auc_scores = {}
fpr, tpr = {}, {}

for i, label in enumerate(attack_labels):
    fpr[label], tpr[label], _ = roc_curve(y_binarized[:, i], y_proba[:, i])  # Compute FPR, TPR
    roc_auc_scores[label] = roc_auc_score(y_binarized[:, i], y_proba[:, i])  # Compute AUC

# Print ROC-AUC scores
print(f"\n---ROC-AUC SCORES---")
for label in attack_labels:
    print(f"ROC-AUC Score (Class {label}: {roc_auc_scores[label]})")

# Plot ROC curves for all classes
plt.figure(figsize=(8, 6))
for label in attack_labels:
    plt.plot(fpr[label], tpr[label], lw=2, label=f'Class {label} (AUC = {roc_auc_scores[label]})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve (Multiclass)')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f"./{results_path}/roc_auc_curve_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()


# Output in a txt file
with open(f'./{results_path}/results_{ml_algo_short}_{feature_count}.txt', 'w') as file:
    # file.write(f"Best Parameters for {ml_algo_short}: {best_params}\n")

    file.write(f"\n---SCORES---")
    file.write(f"\nAccuracy: \n{accuracy_scores}")
    file.write(f"\nAverage: {np.mean(accuracy_scores)}\n")

    file.write(f"\nPrecision: \n{precision_scores}")
    file.write(f"\nAverage: {np.mean(precision_scores)}\n")

    file.write(f"\nRecall: \n{recall_scores}")
    file.write(f"\nAverage: {np.mean(recall_scores)}\n")

    file.write(f"\nF1 Score: \n{f1_scores}")
    file.write(f"\nAverage: {np.mean(f1_scores)}\n")

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
    for label in attack_labels:
        file.write(f"\nROC-AUC Score (Class {label}: {roc_auc_scores[label]})")
    file.write(f"\n")

    file.write(f"\nTime it took to execute (in seconds): {time.time() - start_time:.4f}")

# Save the trained model to be used later
dump(clf, f'multiclass_{ml_algo_short}_{feature_count}_model.joblib')

end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")

