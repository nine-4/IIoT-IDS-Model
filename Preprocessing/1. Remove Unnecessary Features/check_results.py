import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Checking columns...")

print(f"Before: {pd.read_csv('wustl_iiot_2021.csv').columns.tolist()}")
print()
print(f"After: {pd.read_csv('removed_features.csv', low_memory=False).columns.tolist()}")

end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time}")