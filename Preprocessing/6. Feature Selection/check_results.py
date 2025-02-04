import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Checking feature selection results...")

train_41 = pd.read_csv("training_data_41.csv")
test_41 = pd.read_csv("testing_data_41.csv")

train_11 = pd.read_csv("training_data_11.csv")
test_11 = pd.read_csv("testing_data_11.csv")

print(f"training_data_41 features: {train_41.columns.tolist()}")
print(f"testing_data_41 features: {test_41.columns.tolist()}")
print()
print(f"training_data_11 features: {train_11.columns.tolist()}")
print(f"testing_data_11 features: {test_11.columns.tolist()}")


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")