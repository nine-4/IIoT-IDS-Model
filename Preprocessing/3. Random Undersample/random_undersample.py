import time
import pandas as pd
from sklearn.utils import resample

# Tracks execution time
start_time = time.time()

print("Performing undersampling...")

# Read datasets
train_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('testing_data.csv')

# Separate benign and non-benign traffic in training set
benign_train = train_data[train_data['Traffic'] == 'normal']
non_benign_train = train_data[train_data['Traffic'] != 'normal']

# Undersample benign traffic using sklearn
undersampled_benign = resample(
    benign_train, 
    replace=False,
    n_samples=len(non_benign_train), 
    random_state=42
)

# Excess benign samples
excess_benign = benign_train.drop(undersampled_benign.index)

# Create new balanced training set
balanced_train = pd.concat([undersampled_benign, non_benign_train])

# Add excess benign samples to test set
balanced_test = pd.concat([test_data, excess_benign])

# Save updated datasets
balanced_train.to_csv('training_data_undr.csv', index=False)
balanced_test.to_csv('testing_data_undr.csv', index=False)

# Print distributions
print("Training Set Distribution:")
print(balanced_train['Traffic'].value_counts(normalize=True))
print("\nTest Set Distribution:")
print(balanced_test['Traffic'].value_counts(normalize=True))


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time}")