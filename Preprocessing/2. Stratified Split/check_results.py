import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Checking data split...")

df = pd.read_csv('removed_features.csv')
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

print(f"Training set: {train_df.shape[0]} rows")
print(f"Testing set: {test_df.shape[0]} rows")

print("Class distribution in original dataset:")
print(df["Traffic"].value_counts(normalize=True))

print("\nClass distribution in training set:")
print(train_df["Traffic"].value_counts(normalize=True))

print("\nClass distribution in testing set:")
print(test_df["Traffic"].value_counts(normalize=True))


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")

