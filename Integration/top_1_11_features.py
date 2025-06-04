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
binary_ml_algo_short = "RF" # RF
multiclass_ml_algo_short = "KNN" # DT or KNN (RF results in perfect score)

feature_count = 11

binary_model_filename = f"./Models/binary_{binary_ml_algo_short}_{feature_count}_model.joblib"
if multiclass_ml_algo_short == "DT":
    # For DT (10 Features)
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
y_true_binary = df["Target"]     # 0: benign, 1: attack
y_true_multiclass = df["Traffic"]  # Used only for attack instances

# ---LEVEL 1: BINARY CLASSIFICATION--- #
print("Running Binary Classification...")
y_pred_binary = binary_model.predict(X)

# Split based on binary output
attack_indices = np.where(y_pred_binary == 1)[0]
benign_indices = np.where(y_pred_binary == 0)[0]

X_attacks = X.iloc[attack_indices]
true_attack_labels = y_true_multiclass.iloc[attack_indices]  # True labels for attacks

# ---LEVEL 2: MULTICLASS CLASSIFICATION--- #
print("Running Multiclass Classification for Detected Attacks...")

if multiclass_ml_algo_short == "DT":
    X_attacks = X_attacks.drop(columns=["IdleTime"])

y_pred_multiclass = multiclass_model.predict(X_attacks)



# ---EVALUATION--- #
# Binary classification evaluation
binary_acc = accuracy_score(y_true_binary, y_pred_binary)
binary_prec = precision_score(y_true_binary, y_pred_binary)
binary_rec = recall_score(y_true_binary, y_pred_binary)
binary_f1 = f1_score(y_true_binary, y_pred_binary)
binary_cm = confusion_matrix(y_true_binary, y_pred_binary)

# Multiclass evaluation (only on predicted attacks)
mc_acc = accuracy_score(true_attack_labels, y_pred_multiclass)
mc_prec = precision_score(true_attack_labels, y_pred_multiclass, average="weighted", zero_division=True)
mc_rec = recall_score(true_attack_labels, y_pred_multiclass, average="weighted")
mc_f1 = f1_score(true_attack_labels, y_pred_multiclass, average="weighted")

# Get only labels present in the multiclass data
mc_labels = sorted(list(set(true_attack_labels) | set(y_pred_multiclass)))
# Generate confusion matrix
mc_cm = confusion_matrix(true_attack_labels, y_pred_multiclass, labels=mc_labels)

# Save metrics
with open(f"{results_path}/multi_level_results.txt", "w") as f:
    f.write("---LEVEL 1: Binary Classification---\n")
    f.write(f"Accuracy: {binary_acc:.10f}\n")
    f.write(f"Precision: {binary_prec:.10f}\n")
    f.write(f"Recall: {binary_rec:.10f}\n")
    f.write(f"F1 Score: {binary_f1:.10f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(binary_cm, separator=", ") + "\n\n")

    f.write("---LEVEL 2: Multiclass Classification---\n")
    f.write(f"Samples passed to Level 2: {len(attack_indices)}\n")
    f.write(f"Accuracy: {mc_acc:.10f}\n")
    f.write(f"Precision (weighted): {mc_prec:.10f}\n")
    f.write(f"Recall (weighted): {mc_rec:.10f}\n")
    f.write(f"F1 Score (weighted): {mc_f1:.10f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(mc_cm, separator=", ") + "\n")

# ---PLOTTING--- #
# Binary confusion matrix
disp_binary = ConfusionMatrixDisplay(confusion_matrix=binary_cm, display_labels=["Benign", "Attack"])
disp_binary.plot(cmap=plt.cm.Blues)
plt.title("Binary Confusion Matrix")
plt.savefig(f"{results_path}/conf_matrix_binary.png", dpi=300, bbox_inches='tight')
plt.close()

# Multiclass confusion matrix
disp_mc = ConfusionMatrixDisplay(confusion_matrix=mc_cm, display_labels=mc_labels)
disp_mc.plot(cmap=plt.cm.Greens)
plt.title("Multiclass Confusion Matrix (Attacks Only)")
plt.savefig(f"{results_path}/conf_matrix_multiclass.png", dpi=300, bbox_inches='tight')
plt.close()

# ---SUMMARY--- #
end_time = time.time()
print(f"\nTotal samples: {len(df)}")
print(f"Attacks detected by binary model: {len(attack_indices)}")
print()
print(f"Binary Accuracy: {binary_acc:.10f}")
print(f"Binary Precision: {binary_prec:.10f}")
print(f"Binary Recall: {binary_rec:.10f}")
print(f"Binary F1: {binary_f1:.10f}")
print()
print(f"Multiclass Accuracy: {mc_acc:.10f}")
print(f"Multiclass Precision: {mc_prec:.10f}")
print(f"Multiclass Recall: {mc_rec:.10f}")
print(f"Multiclass F1: {mc_f1:.10f}")
print(f"\nExecution Time: {end_time - start_time:.10f} seconds")
