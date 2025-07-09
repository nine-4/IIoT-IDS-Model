import shap
import pandas as pd
import numpy as np
import joblib  # or use pickle

# === Configuration ===
model_path = "multiclass_RF_41_model.joblib"  # Change to your actual model path
data_path = "training_data_41.csv"  # Your test feature data
class_names = ['Backdoor', 'CommInj', 'DoS', 'Reconn', 'normal']


# === Step 1: Load model and data ===
model = joblib.load(model_path)
X_raw = pd.read_csv(data_path)  # Shape: [n_samples, n_features]

X = X_raw.drop(columns=['Traffic', 'Target'], errors='ignore')

# === Step 2: Create SHAP Explainer ===
# Use predict_proba for multiclass
explainer = shap.Explainer(model.predict_proba, X)
shap_values = explainer(X)  # Shape: [n_samples, n_classes, n_features]

# === Step 3: Generate SHAP summary for each class ===
summary_df = pd.DataFrame({"Feature": X.columns})

for class_index, class_name in enumerate(class_names):
    # Get absolute mean SHAP value per feature for this class
    mean_shap = np.abs(shap_values[:, class_index, :].values).mean(axis=0)

    # Add to DataFrame
    summary_df[f"{class_name}_SHAP"] = mean_shap
    summary_df[f"{class_name}_Rank"] = summary_df[f"{class_name}_SHAP"].rank(ascending=False)

# === Step 4: Sort and display top features for each class ===
for class_name in class_names:
    print(f"\nTop 10 Features for {class_name}")
    print(summary_df[['Feature', f"{class_name}_SHAP", f"{class_name}_Rank"]]
          .sort_values(f"{class_name}_Rank")
          .head(10))

# === Optional: Plot SHAP bar chart for one class ===
# Example: Top features for 'Backdoor' class (index 0)
shap.plots.bar(shap_values[:, 0, :], max_display=15)