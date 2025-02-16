import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score)

#-----VARIABLES TO MODIFY-----#
feature_count = 11
ml_algo = "Logistic Regression"
ml_algo_short = "LR"

clf = LogisticRegression(random_state=42, max_iter=1000)
k_folds = 5  # Number of folds
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

# Perform k-fold cross-validation and calculate metrics
accuracy_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(clf, X, y, cv=kf, scoring='precision_weighted')
recall_scores = cross_val_score(clf, X, y, cv=kf, scoring='recall_weighted')
f1_scores = cross_val_score(clf, X, y, cv=kf, scoring='f1_weighted')

# Output the average scores across all folds
print(f"\n---CROSS VALIDATION SCORES---")
print(f"ACCURACY:\n{accuracy_scores}")
print(f"Average Accuracy: {accuracy_scores.mean()}")

print(f"\nPRECISION:\n{precision_scores}")
print(f"Average Precision: {precision_scores.mean()}")

print(f"\nRECALL:\n{recall_scores}")
print(f"Average Recall: {recall_scores.mean()}")

print(f"\nF1 SCORE:\n{f1_scores}")
print(f"Average F1 Score: {f1_scores.mean()}")


# Get predicted values using cross-validation
y_pred = cross_val_predict(clf, X, y, cv=kf)
y_proba = cross_val_predict(clf, X, y, cv=kf, method="predict_proba")[:, 1]  # Probabilities for class 1

# Compute confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
normalized_conf_matrix = np.round(conf_matrix / np.sum(conf_matrix, axis=1).reshape(-1, 1), decimals=5)

print(f"\n---CROSS VALIDATION PREDICTIONS---")
print("Confusion Matrix:")
print(conf_matrix)
print("\nNormalized Confusion Matrix:")
print(normalized_conf_matrix)

conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1])
conf_matrix_display.plot(cmap=plt.cm.Blues)
plt.savefig(f"conf_matrix_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

normalized_conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = normalized_conf_matrix, display_labels = [0, 1])
normalized_conf_matrix_display.plot(cmap=plt.cm.Greens)
plt.savefig(f"conf_matrix_normalized_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()

accuracy_pred = accuracy_score(y, y_pred)
precision_pred_1 = precision_score(y, y_pred, pos_label=1)
precision_pred_0 = precision_score(y, y_pred, pos_label=0)
recall_pred_1 = recall_score(y, y_pred, pos_label=1)
recall_pred_0 = recall_score(y, y_pred, pos_label=0)
f1_pred_1 = f1_score(y, y_pred, pos_label=1)
f1_pred_0 = f1_score(y, y_pred, pos_label=0)

# Compute ROC-AUC score
roc_auc = roc_auc_score(y, y_proba)
fpr, tpr, _ = roc_curve(y, y_proba)

# Print ROC-AUC score
print(f"\nROC-AUC Score: {roc_auc}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f"roc_auc_curve_{feature_count}.png", dpi=300, bbox_inches='tight')
plt.close()


print(f"\nAccuracy: {accuracy_pred}")

print(f"\nPrecision (Class 1): {precision_pred_1}")
print(f"Precision (Class 0): {precision_pred_0}")

print(f"\nRecall (Class 1): {recall_pred_1}")
print(f"Recall (Class 0): {recall_pred_0}")

print(f"\nF1 Score (Class 1): {f1_pred_1}")
print(f"F1 Score (Class 0): {f1_pred_0}")

# Output in a txt file
with open(f'results_{ml_algo_short}_{feature_count}.txt', 'w') as file:
    file.write(f"\n---CROSS VALIDATION SCORES---")
    file.write(f"\nACCURACY:\n{accuracy_scores}")
    file.write(f"\nAverage Accuracy: {accuracy_scores.mean()}\n")

    file.write(f"\nPRECISION:\n{precision_scores}")
    file.write(f"\nAverage Precision: {precision_scores.mean()}\n")

    file.write(f"\nRECALL:\n{recall_scores}")
    file.write(f"\nAverage Recall: {recall_scores.mean()}\n")

    file.write(f"\nF1 SCORE:\n{f1_scores}")
    file.write(f"\nAverage F1 Score: {f1_scores.mean()}\n")

    file.write(f"\n---CROSS VALIDATION PREDICTIONS---")
    file.write("\nConfusion Matrix:\n")
    file.write(f"{conf_matrix}\n")

    file.write("\nNormalized Confusion Matrix:\n")
    file.write(f"{normalized_conf_matrix}\n")

    file.write(f"\nAccuracy: {accuracy_pred}\n")

    file.write(f"\nPrecision (Class 1): {precision_pred_1}\n")
    file.write(f"Precision (Class 0): {precision_pred_0}\n")

    file.write(f"\nRecall (Class 1): {recall_pred_1}\n")
    file.write(f"Recall (Class 0): {recall_pred_0}\n")

    file.write(f"\nF1 Score (Class 1): {f1_pred_1}\n")
    file.write(f"F1 Score (Class 0): {f1_pred_0}\n")

    file.write(f"\nROC-AUC Score: {roc_auc}\n")

    file.write(f"\nTime it took to execute (in seconds): {time.time() - start_time:.4f}")


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")

