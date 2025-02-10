import time
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# Tracks execution time
start_time = time.time()

print("Training Random Forest using 11 Features...")

# Load the dataset
df = pd.read_csv("../../training_data_11.csv")

# Split dataset into features (X) and target variable (y)
X = df.drop(columns=["Traffic", "Target"])
y = df["Target"]

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Set up k-fold cross-validation
k_folds = 5  # Number of folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Perform k-fold cross-validation and calculate metrics
accuracy_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(clf, X, y, cv=kf, scoring='precision_weighted')
recall_scores = cross_val_score(clf, X, y, cv=kf, scoring='recall_weighted')
f1_scores = cross_val_score(clf, X, y, cv=kf, scoring='f1_weighted')

# Output the average scores across all folds
print(f"\nACCURACY:\n{accuracy_scores}")
print(f"Average Accuracy: {accuracy_scores.mean()}")

print(f"\nPRECISION:\n{precision_scores}")
print(f"Average Precision: {precision_scores.mean()}")

print(f"\nRECALL:\n{recall_scores}")
print(f"Average Recall: {recall_scores.mean()}")

print(f"\nF1 SCORE:\n{f1_scores}")
print(f"Average F1 Score: {f1_scores.mean()}")

# Output in a txt file
with open('results_RF_11.txt', 'w') as file:
    file.write(f"\nACCURACY:\n{accuracy_scores}")
    file.write(f"\nAverage Accuracy: {accuracy_scores.mean()}\n")

    file.write(f"\nPRECISION:\n{precision_scores}")
    file.write(f"\nAverage Precision: {precision_scores.mean()}\n")

    file.write(f"\nRECALL:\n{recall_scores}")
    file.write(f"\nAverage Recall: {recall_scores.mean()}\n")

    file.write(f"\nF1 SCORE:\n{f1_scores}")
    file.write(f"\nAverage F1 Score: {f1_scores.mean()}\n")


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")

