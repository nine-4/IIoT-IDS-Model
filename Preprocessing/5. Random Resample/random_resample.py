import time
import pandas as pd
from sklearn.utils import resample

# Tracks execution time
start_time = time.time()

print("Performing undersampling and oversampling...")

# Read datasets
train_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('testing_data.csv')

# Separate traffic types in the training set
normal_traffic = train_data[train_data['Traffic'] == 'normal']
dos_traffic = train_data[train_data['Traffic'] == 'DoS']
reconn_traffic = train_data[train_data['Traffic'] == 'Reconn']
comminj_traffic = train_data[train_data['Traffic'] == 'CommInj']
backdoor_traffic = train_data[train_data['Traffic'] == 'Backdoor']

# Define the desired sample sizes
total_training_samples = len(train_data)
normal_samples = int(0.50 * total_training_samples)  # 50% normal
attack_samples = int(0.125 * total_training_samples)  # 12.5% per attack type

# Undersample normal traffic
undersampled_normal = resample(
    normal_traffic,
    replace=False,  # Undersampling
    n_samples=normal_samples,
    random_state=42
)

# Takes note of the excess normal samples to move to the test set
excess_normal = normal_traffic.drop(undersampled_normal.index)

# Oversample each attack type
oversampled_dos = resample(
    dos_traffic,
    replace=True,  # Oversampling
    n_samples=attack_samples,
    random_state=42
)

oversampled_reconn = resample(
    reconn_traffic,
    replace=True,  # Oversampling
    n_samples=attack_samples,
    random_state=42
)

oversampled_comminj = resample(
    comminj_traffic,
    replace=True,  # Oversampling
    n_samples=attack_samples,
    random_state=42
)

oversampled_backdoor = resample(
    backdoor_traffic,
    replace=True,  # Oversampling
    n_samples=attack_samples,
    random_state=42
)

# Combine undersampled and oversampled data to create a balanced training set
train_balanced = pd.concat([
    undersampled_normal,
    oversampled_dos,
    oversampled_reconn,
    oversampled_comminj,
    oversampled_backdoor
])

# Add excess normal samples to the test set
test_updated = pd.concat([test_data, excess_normal])

# Save updated datasets
train_balanced.to_csv('training_data_41.csv', index=False)
test_updated.to_csv('testing_data_41.csv', index=False)

# Check number of rows
print(f"Training set: {train_balanced.shape[0]} rows")
print(f"Testing set: {test_updated.shape[0]} rows")

# Print distributions
print("Training Set Distribution:")
print(train_balanced['Traffic'].value_counts(normalize=True))
print("\nTest Set Distribution:")
print(test_updated['Traffic'].value_counts(normalize=True))

end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")
