import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# === Load the Random Forest Model ===
model = joblib.load("binary_RF_11_model.joblib")  # Replace with your filename

# === Choose which tree to visualize ===
tree_index = 49  # You can change this to visualize other trees

# === Extract the tree estimator ===
estimator = model.estimators_[tree_index]

# === Plot the tree using matplotlib ===
plt.figure(figsize=(20, 10))  # Adjust size as needed
plot_tree(estimator,
          filled=True,
          rounded=True,
          feature_names=getattr(model, "feature_names_in_", None),
          class_names=[str(cls) for cls in model.classes_] if hasattr(model, 'classes_') else None)
plt.title(f"Random Forest Tree #{tree_index}")
plt.savefig("random_forest_tree_1.png", bbox_inches='tight', dpi=300)  # Adjust filename and dpi as needed
plt.show()