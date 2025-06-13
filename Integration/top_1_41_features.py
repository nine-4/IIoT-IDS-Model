import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# -----CONFIGURATION----- #
binary_ml_algo_short = "RF"  # RF
multiclass_ml_algo_short = "KNN"  # DT or KNN (RF results in perfect score)

feature_count = 41

binary_model_filename = f"./Models/binary_{binary_ml_algo_short}_{feature_count}_model.joblib"
if multiclass_ml_algo_short == "DT":
    multiclass_model_filename = f"./Models/multiclass_DT_10_model.joblib"
else:
    multiclass_model_filename = f"./Models/multiclass_{multiclass_ml_algo_short}_{feature_count}_model.joblib"

unseen_csv = f"./Data/testing_data_{feature_count}.csv"
results_path = f"results_{feature_count}"
os.makedirs(results_path, exist_ok=True)
# ------------------------ #

start_time = time.time()

# Load Models
print("Loading Models...")
binary_model = load(binary_model_filename)
multiclass_model = load(multiclass_model_filename)

# Load unseen data
print("Loading unseen dataset...")
df = pd.read_csv(unseen_csv)

X = df.drop(columns=["Traffic", "Target"])
y_true_binary = df["Target"]  # 0: benign, 1: attack
y_true_multiclass = df["Traffic"]  # Used only for attack instances

# ---LEVEL 1: BINARY CLASSIFICATION--- #
print("Running Binary Classification...")
y_pred_binary = binary_model.predict(X)

attack_indices = np.where(y_pred_binary == 1)[0]
benign_indices = np.where(y_pred_binary == 0)[0]

X_attacks = X.iloc[attack_indices]
true_attack_labels = y_true_multiclass.iloc[attack_indices]

# ---LEVEL 2: MULTICLASS CLASSIFICATION--- #
print("Running Multiclass Classification for Detected Attacks...")
if multiclass_ml_algo_short == "DT":
    X_attacks = X_attacks.drop(columns=["IdleTime"])

y_pred_multiclass = multiclass_model.predict(X_attacks)

print("Calculating evaluations...")
# ---COMBINED EVALUATION--- #
y_combined_true = []
y_combined_pred = []

for i in range(len(df)):
    if y_pred_binary[i] == 0:
        y_combined_true.append("normal")
        y_combined_pred.append("normal")
    else:
        y_combined_true.append(y_true_multiclass.iloc[i])
        y_combined_pred.append(y_pred_multiclass[attack_indices.tolist().index(i)])

combined_labels = sorted(list(set(y_combined_true) | set(y_combined_pred)))
combined_cm = confusion_matrix(y_combined_true, y_combined_pred, labels=combined_labels)

combined_acc = accuracy_score(y_combined_true, y_combined_pred)
combined_prec = precision_score(y_combined_true, y_combined_pred, average="weighted", zero_division=True)
combined_rec = recall_score(y_combined_true, y_combined_pred, average="weighted")
combined_f1 = f1_score(y_combined_true, y_combined_pred, average="weighted")
# Gets metrics per class
combined_class_scores = {}
for cls in combined_labels:
    p = precision_score(y_combined_true, y_combined_pred, labels=[cls], average="macro", zero_division=True)
    r = recall_score(y_combined_true, y_combined_pred, labels=[cls], average="macro", zero_division=True)
    f = f1_score(y_combined_true, y_combined_pred, labels=[cls], average="macro", zero_division=True)
    combined_class_scores[cls] = {"precision": p, "recall": r, "f1": f}

# ---PER LEVEL METRICS--- #
binary_acc = accuracy_score(y_true_binary, y_pred_binary)
binary_prec_0 = precision_score(y_true_binary, y_pred_binary, pos_label=0)
binary_prec_1 = precision_score(y_true_binary, y_pred_binary, pos_label=1)
binary_rec_0 = recall_score(y_true_binary, y_pred_binary, pos_label=0)
binary_rec_1 = recall_score(y_true_binary, y_pred_binary, pos_label=1)
binary_f1_0 = f1_score(y_true_binary, y_pred_binary, pos_label=0)
binary_f1_1 = f1_score(y_true_binary, y_pred_binary, pos_label=1)
binary_cm = confusion_matrix(y_true_binary, y_pred_binary)

mc_acc = accuracy_score(true_attack_labels, y_pred_multiclass)
mc_prec = precision_score(true_attack_labels, y_pred_multiclass, average="weighted", zero_division=True)
mc_rec = recall_score(true_attack_labels, y_pred_multiclass, average="weighted")
mc_f1 = f1_score(true_attack_labels, y_pred_multiclass, average="weighted")
mc_labels = sorted(list(set(true_attack_labels) | set(y_pred_multiclass)))
mc_class_scores = {}
# Gets metrics per class
for cls in mc_labels:
    p = precision_score(true_attack_labels, y_pred_multiclass, labels=[cls], average="macro", zero_division=True)
    r = recall_score(true_attack_labels, y_pred_multiclass, labels=[cls], average="macro", zero_division=True)
    f = f1_score(true_attack_labels, y_pred_multiclass, labels=[cls], average="macro", zero_division=True)
    mc_class_scores[cls] = {"precision": p, "recall": r, "f1": f}

mc_cm = confusion_matrix(true_attack_labels, y_pred_multiclass, labels=mc_labels)

# ---SAVE RESULTS--- #
with open(f"{results_path}/multi_level_results.txt", "w") as f:
    f.write("---LEVEL 1: Binary Classification---\n")
    f.write(f"Accuracy: {binary_acc:.10f}\n")
    f.write(f"Class 0 (normal): Precision={binary_prec_0:.10f}, Recall={binary_rec_0:.10f}, F1={binary_f1_0:.10f}\n")
    f.write(f"Class 1 (attack): Precision={binary_prec_1:.10f}, Recall={binary_rec_1:.10f}, F1={binary_f1_1:.10f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(binary_cm, separator=", ") + "\n\n")

    f.write("---LEVEL 2: Multiclass Classification---\n")
    f.write(f"Samples passed to Level 2: {len(attack_indices)}\n")
    f.write(f"Accuracy: {mc_acc:.10f}\n")
    f.write(f"Precision (weighted): {mc_prec:.10f}\n")
    f.write(f"Recall (weighted): {mc_rec:.10f}\n")
    f.write(f"F1 Score (weighted): {mc_f1:.10f}\n")

    f.write("Per-class Metrics:\n")
    for cls, scores in mc_class_scores.items():
        f.write(
            f"Class '{cls}': Precision={scores['precision']:.10f}, Recall={scores['recall']:.10f}, F1={scores['f1']:.10f}\n")

    f.write("Confusion Matrix:\n")
    f.write(np.array2string(mc_cm, separator=", ") + "\n\n")

    f.write("---OVERALL SYSTEM EVALUATION---\n")
    f.write(f"Accuracy: {combined_acc:.10f}\n")
    f.write(f"Precision (weighted): {combined_prec:.10f}\n")
    f.write(f"Recall (weighted): {combined_rec:.10f}\n")
    f.write(f"F1 Score (weighted): {combined_f1:.10f}\n")

    f.write("Per-class Metrics:\n")
    for cls, scores in combined_class_scores.items():
        f.write(
            f"Class '{cls}': Precision={scores['precision']:.10f}, Recall={scores['recall']:.10f}, F1={scores['f1']:.10f}\n")

    f.write("Confusion Matrix:\n")
    f.write(np.array2string(combined_cm, separator=", ") + "\n")

# ---PLOTTING CONFUSION MATRICES--- #
disp_binary = ConfusionMatrixDisplay(confusion_matrix=binary_cm, display_labels=["normal", "attack"])
disp_binary.plot(cmap=plt.cm.Blues)
plt.title("Binary Confusion Matrix")
plt.savefig(f"{results_path}/conf_matrix_binary.png", dpi=300, bbox_inches='tight')
plt.close()

disp_mc = ConfusionMatrixDisplay(confusion_matrix=mc_cm, display_labels=mc_labels)
disp_mc.plot(cmap=plt.cm.Greens)
plt.title("Multiclass Confusion Matrix (Attacks Only)")
plt.savefig(f"{results_path}/conf_matrix_multiclass.png", dpi=300, bbox_inches='tight')
plt.close()

disp_combined = ConfusionMatrixDisplay(confusion_matrix=combined_cm, display_labels=combined_labels)
disp_combined.plot(cmap=plt.cm.Purples)
plt.title("Overall System Confusion Matrix")
plt.savefig(f"{results_path}/conf_matrix_combined.png", dpi=300, bbox_inches='tight')
plt.close()

# ---SUMMARY OUTPUT--- #
end_time = time.time()
print(f"\nTotal samples: {len(df)}")
print(f"Attacks detected by binary model: {len(attack_indices)}")
print()
print(f"Binary Accuracy: {binary_acc:.10f}")
print(f"Binary Class 0 (normal) - Precision: {binary_prec_0:.10f}, Recall: {binary_rec_0:.10f}, F1: {binary_f1_0:.10f}")
print(f"Binary Class 1 (attack) - Precision: {binary_prec_1:.10f}, Recall: {binary_rec_1:.10f}, F1: {binary_f1_1:.10f}")
print()
print(f"Multiclass Accuracy: {mc_acc:.10f}")
print("Multiclass Per-Class Metrics:")
for cls, scores in mc_class_scores.items():
    print(f"  {cls}: Precision={scores['precision']:.10f}, Recall={scores['recall']:.10f}, F1={scores['f1']:.10f}")
print()
print(f"Combined Accuracy: {combined_acc:.10f}")
print("Combined Per-Class Metrics:")
for cls, scores in combined_class_scores.items():
    print(f"  {cls}: Precision={scores['precision']:.10f}, Recall={scores['recall']:.10f}, F1={scores['f1']:.10f}")
print(f"\nExecution Time: {end_time - start_time:.10f} seconds")
