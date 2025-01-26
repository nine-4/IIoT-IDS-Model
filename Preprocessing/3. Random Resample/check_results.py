import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Checking undersampling and oversampling results...")

train = pd.read_csv("training_data.csv")
test = pd.read_csv("testing_data.csv")

train_balanced = pd.read_csv("training_data_41.csv")
test_updated = pd.read_csv("testing_data_41.csv")

# Check number of rows
print("Original Sets")
print(f"Training set: {train.shape[0]} rows")
print(f"Testing set: {test.shape[0]} rows")
print()
print("Updated Sets")
print(f"Training set: {train_balanced.shape[0]} rows")
print(f"Testing set: {test_updated.shape[0]} rows")
print()

# Print distributions
print("Training Set Distribution:")
print(train_balanced['Traffic'].value_counts(normalize=True))
print("\nTest Set Distribution:")
print(test_updated['Traffic'].value_counts(normalize=True))


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")