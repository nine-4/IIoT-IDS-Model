import time
import pandas as pd

start_time = time.time()

print("Removing duplicate samples...")

input_file = 'removed_features.csv'
output_file = 'no_duplicates.csv'
chunksize = 10_000

seen = set()  # To track seen rows as tuples
total_rows = 0
total_unique = 0

for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
    mask = []
    for row in chunk.itertuples(index=False, name=None):
        if row in seen:
            mask.append(False)  # Duplicate row
        else:
            seen.add(row)
            mask.append(True)  # Unique row

    unique_chunk = chunk[mask]

    unique_chunk.to_csv(output_file, mode='a', index=False, header=(i == 0))

removed = total_rows - total_unique
print(f"Number of samples removed: {removed}")
print(f"Total number of samples left: {total_unique}")

end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")