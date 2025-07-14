import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import classification_report
from sklearn.feature_selection import f_classif, mutual_info_classif

# ---------- SETUP ---------- #
feature_count = 41
model_path = f"./Models/binary_RF_{feature_count}_model.joblib"
data_path = f"./Data/testing_data_{feature_count}.csv"
output_path = f"./results_binary_RF_{feature_count}"
os.makedirs(output_path, exist_ok=True)

# ---------- LOAD DATA ---------- #
df = pd.read_csv(data_path)
X = df.drop(columns=["Traffic", "Target"])
y = df["Target"]  # 0 = normal, 1 = attack

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
plt.title("Top Features by F-Score (Binary)")
plt.tight_layout()
plt.savefig(f"{output_path}/feature_importance_fscore.png", dpi=300)
plt.close()

# ---------- ANALYZE MISCLASSIFIED ---------- #
top_feats = importance_df["Feature"].head(5)

for feat in top_feats:
    plt.figure(figsize=(10, 4))
    sns.kdeplot(X[~misclassified][feat], label="Correctly Classified", fill=True)
    sns.kdeplot(X[misclassified][feat], label="Misclassified", fill=True)
    plt.title(f"Distribution of '{feat}' (Correct vs Misclassified)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/dist_{feat}_misclass.png", dpi=300)
    plt.close()

# ---------- EXPORT MISCLASSIFIED ---------- #
df[misclassified].to_csv(f"{output_path}/misclassified_samples.csv", index=False)

# ---------- CLASSIFICATION REPORT ---------- #
report = classification_report(y, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{output_path}/classification_report.csv")

print(f"Analysis complete. Results saved in: {output_path}")
