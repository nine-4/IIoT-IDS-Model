import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Checking One Hot Encoding results...")

print(f"Columns: {pd.read_csv('one_hot_enc.csv').columns.tolist()}")


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")