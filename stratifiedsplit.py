import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('removedfeatures.csv')

class_column = 'Traffic'  

# Perform a stratified split (70% training, 30% testing)
train_df, test_df = train_test_split(
    df,
    test_size=0.3,  # 30% for testing
    stratify=df[class_column],  # Ensure stratified split
    random_state=42  # For reproducibility
)

# Save the training and testing sets to separate files
train_df.to_csv('training_data.csv', index=False)
test_df.to_csv('testing_data.csv', index=False)

print(f"Training set: {train_df.shape[0]} rows")
print(f"Testing set: {test_df.shape[0]} rows")

print("Class distribution in original dataset:")
print(df[class_column].value_counts(normalize=True))

print("\nClass distribution in training set:")
print(train_df[class_column].value_counts(normalize=True))

print("\nClass distribution in testing set:")
print(test_df[class_column].value_counts(normalize=True))