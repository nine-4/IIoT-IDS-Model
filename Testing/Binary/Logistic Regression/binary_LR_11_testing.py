import time
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from joblib import load
import matplotlib.pyplot as plt
import os

# -----VARIABLES TO MODIFY----- #
feature_count = 11
ml_algo = "Logistic Regression"
ml_algo_short = "LR"

model_filename = f"binary_{ml_algo_short}_{feature_count}_model.joblib"
unseen_csv = f"../../testing_data_{feature_count}.csv"

results_path = f"testing_results_{feature_count}"
os.makedirs(results_path, exist_ok=True)
# ----------------------------- #

start_time = time.time()

print(f"Loading trained model: {model_filename}")
model = load(model_filename)

print(f"Loading unseen data: {unseen_csv}")
df = pd.read_csv(unseen_csv)

# Split dataset into features (X) and target variable (y)
X_unseen = df.drop(columns=["Traffic", "Target"])
y_true = df["Target"]

print("Predicting on unseen data...")
y_pred = model.predict(X_unseen)
y_proba = model.predict_proba(X_unseen)[:, 1]

# --- Metrics ---
accuracy = accuracy_score(y_true, y_pred)
precision_1 = precision_score(y_true, y_pred, pos_label=1)
precision_0 = precision_score(y_true, y_pred, pos_label=0)
recall_1 = recall_score(y_true, y_pred, pos_label=1)
recall_0 = recall_score(y_true, y_pred, pos_label=0)
f1_1 = f1_score(y_true, y_pred, pos_label=1)
f1_0 = f1_score(y_true, y_pred, pos_label=0)
roc_auc = roc_auc_score(y_true, y_proba)
cm = confusion_matrix(y_true, y_pred)

print("\n---TEST METRICS---")
print(f"Accuracy: {accuracy:.4f}")
print()
print(f"Precision (Class 1): {precision_1:.4f}")
print(f"Precision (Class 0): {precision_0:.4f}")
print()
print(f"Recall (Class 1): {recall_1:.4f}")
print(f"Recall (Class 0): {recall_0:.4f}")
print()
print(f"F1 Score (Class 1): {f1_1:.4f}")
print(f"F1 Score (Class 0): {f1_0:.4f}")
print()
print(f"ROC-AUC: {roc_auc:.4f}")
print()
print("Confusion Matrix:")
print(cm)

# Save metrics to file
with open(f"{results_path}/test_results_{ml_algo_short}_{feature_count}.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(f"Precision (Class 1): {precision_1:.4f}\n")
    f.write(f"Precision (Class 0): {precision_0:.4f}\n\n")
    f.write(f"Recall (Class 1): {recall_1:.4f}\n")
    f.write(f"Recall (Class 0): {recall_0:.4f}\n\n")
    f.write(f"F1 Score (Class 1): {f1_1:.4f}\n")
    f.write(f"F1 Score (Class 0): {f1_0:.4f}\n\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Unseen Data")
plt.savefig(f"{results_path}/conf_matrix_{feature_count}_testing.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Unseen Data")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(f"{results_path}/roc_auc_curve_{ml_algo_short}_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

end_time = time.time()
print(f"\nTesting completed in {end_time - start_time:.2f} seconds.")
