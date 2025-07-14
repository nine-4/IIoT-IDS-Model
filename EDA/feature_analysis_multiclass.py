import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import classification_report
from sklearn.feature_selection import f_classif, mutual_info_classif

# ---------- SETUP ---------- #
ml_algo_short = "DT"
feature_count = 41

if ml_algo_short == "DT":
    model_path = f"./Models/multiclass_DT_{feature_count - 1}_model.joblib"
    output_path = f"./results_multiclass_DT_{feature_count - 1}"
else:
    model_path = f"./Models/multiclass_{ml_algo_short}_{feature_count}_model.joblib"
    output_path = f"./results_multiclass_{ml_algo_short}_{feature_count}"

data_path = f"./Data/testing_data_{feature_count}.csv"
os.makedirs(output_path, exist_ok=True)

# ---------- LOAD DATA ---------- #
df = pd.read_csv(data_path)

# Important: Remove normal samples (assumed to have Target == 0)
df = df[df["Target"] != 0]

if ml_algo_short == "DT":
    df = df.drop(columns=["IdleTime"])

X = df.drop(columns=["Traffic", "Target"])
y = df["Traffic"]  # Multiclass labels: Backdoor, CommInj, DoS, Reconn

# ---------- LOAD MODEL ---------- #
model = load(model_path)

# ---------- PREDICTIONS ---------- #
y_pred = model.predict(X)
misclassified = y != y_pred

# ---------- FEATURE IMPORTANCE ---------- #
f_scores, _ = f_classif(X, y)
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "F-Score": f_scores,
    "Mutual_Info": mi_scores
}).sort_values(by="F-Score", ascending=False)

importance_df.to_csv(f"{output_path}/feature_importance.csv", index=False)

# ---------- PLOT FEATURE IMPORTANCE ---------- #
plt.figure(figsize=(10, 6))
sns.barplot(x="F-Score", y="Feature", data=importance_df, palette="Blues_d")
plt.title("Top Features by F-Score (Multiclass)")
plt.tight_layout()
plt.savefig(f"{output_path}/feature_importance_fscore.png", dpi=300)
plt.close()

# ---------- ANALYZE MISCLASSIFIED ---------- #
top_feats = importance_df["Feature"].head(5)

# Per-class analysis
classes = sorted(y.unique())
for cls in classes:
    cls_mask = y == cls
    misclass_mask = misclassified & cls_mask
    correct_mask = ~misclassified & cls_mask

    for feat in top_feats:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(X[correct_mask][feat], label="Correctly Classified", fill=True)
        sns.kdeplot(X[misclass_mask][feat], label="Misclassified", fill=True)
        plt.title(f"{cls} - '{feat}' Distribution (Correct vs Misclassified)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/dist_{cls}_{feat}_misclass.png", dpi=300)
        plt.close()

# ---------- EXPORT MISCLASSIFIED ---------- #
df[misclassified].to_csv(f"{output_path}/misclassified_samples.csv", index=False)

# ---------- CLASSIFICATION REPORT ---------- #
report = classification_report(y, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{output_path}/classification_report.csv")

print(f"Multiclass analysis complete. Results saved in: {output_path}")
