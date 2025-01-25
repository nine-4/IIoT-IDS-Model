import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Removing unnecessary features...")

# Specify the columns to remove
columns_to_remove = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId']

# Read and process the file in chunks
chunksize = 10_000  # Number of rows to process at a time
output_file = 'removed_features.csv'  # Output CSV file

for i, chunk in enumerate(pd.read_csv('wustl_iiot_2021.csv', chunksize=chunksize)):
    # Drop the specified columns
    chunk = chunk.drop(columns=columns_to_remove, errors='ignore')
    
    # Append each chunk to the CSV
    chunk.to_csv(output_file, mode='a', index=False, header=(i == 0))  # Write header only for the first chunk

end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time}")