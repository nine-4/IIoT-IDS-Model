import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Creating train and test sets with Alani's 11 features...")

# Load the original train and test sets
train_41 = pd.read_csv("training_data_41.csv")
test_41 = pd.read_csv("testing_data_41.csv")

# Alani's 11 features
selected_features = ['DIntPkt', 'sTtl', 'Dport', 'IdleTime', 'SIntPkt',
                     'DstBytes', 'pLoss', 'SrcLoad', 'SrcPkts', 'Load', 'DstLoad']


# Retain only the selected features
train_11 = train_41[selected_features]
test_11 = test_41[selected_features]

train_11.to_csv("training_data_11.csv", index=False)
test_11.to_csv("testing_data_11.csv", index=False)

print(f"training_data_11 features: {train_11.columns.tolist()}")
print(f"testing_data_11 features: {test_11.columns.tolist()}")

end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")
