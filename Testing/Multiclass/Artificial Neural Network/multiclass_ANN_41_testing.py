import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from joblib import load

# -----CONFIGURATION----- #
feature_count = 41
ml_algo = "Artificial Neural Network"
ml_algo_short = "ANN"

model_filename = f"multiclass_{ml_algo_short}_{feature_count}_model.joblib"
unseen_csv = f"../../testing_data_{feature_count}.csv"

results_path = f"testing_results_{feature_count}"
os.makedirs(results_path, exist_ok=True)
# ------------------------ #

start_time = time.time()

print(f"Loading trained model: {model_filename}")
model = load(model_filename)

print(f"Loading unseen data: {unseen_csv}")
df = pd.read_csv(unseen_csv)

# Important: remove the "normal" Traffic samples
df = df[df["Target"] != 0]

# Assumes unseen data contains ground-truth labels
X_unseen = df.drop(columns=["Traffic", "Target"])
y_true = df["Traffic"]  # Use Traffic column for multiclass

print("Predicting on unseen data...")
y_pred = model.predict(X_unseen)
y_proba = model.predict_proba(X_unseen)

# Get unique class labels from the model
class_labels = model.classes_

# Evaluation metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted")
rec = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)

print("\n---TEST METRICS---")
print(f"Accuracy: {acc:.4f}")
print()
print(f"Precision (weighted): {prec:.4f}")
print()
print(f"Recall (weighted): {rec:.4f}")
print()
print(f"F1 Score (weighted): {f1:.4f}")
print()
print("Confusion Matrix:")
print(cm)

# Save metrics
with open(f"{results_path}/test_results_{ml_algo_short}_{feature_count}.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision (weighted): {prec:.4f}\n")
    f.write(f"Recall (weighted): {rec:.4f}\n")
    f.write(f"F1 Score (weighted): {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm, separator=', ') + "\n")

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Unseen Data")
plt.savefig(f"{results_path}/conf_matrix_{ml_algo_short}_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

# ROC-AUC per class
y_true_binarized = label_binarize(y_true, classes=class_labels)
fpr, tpr, roc_auc = {}, {}, {}

for i, label in enumerate(class_labels):
    fpr[label], tpr[label], _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
    roc_auc[label] = roc_auc_score(y_true_binarized[:, i], y_proba[:, i])

print("\n---ROC-AUC SCORES---")
for label in class_labels:
    print(f"Class {label}: AUC = {roc_auc[label]:.4f}")

# Save ROC-AUC scores
with open(f"{results_path}/roc_auc_scores.txt", "w") as f:
    for label in class_labels:
        f.write(f"Class {label}: AUC = {roc_auc[label]:.4f}\n")

# Plot ROC curves
plt.figure(figsize=(8, 6))
for label in class_labels:
    plt.plot(fpr[label], tpr[label], lw=2, label=f'Class {label} (AUC = {roc_auc[label]:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve - Unseen Data (Multiclass)")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(f"{results_path}/roc_auc_curve_{ml_algo_short}_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

end_time = time.time()
print(f"\nTesting completed in {end_time - start_time:.2f} seconds.")
