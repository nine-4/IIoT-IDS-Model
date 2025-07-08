import pandas as pd
import os

# --- CONFIGURATION --- #
input_csv = "EDA/normalized.csv"   # Adjust this path to where your file is
output_dir = "EDA/separated"

# --- PRINT CURRENT DIRECTORY (optional debug) --- #
print("Running script from:", os.getcwd())

# --- ENSURE OUTPUT DIRECTORY EXISTS --- #
os.makedirs(output_dir, exist_ok=True)

# --- LOAD DATASET --- #
print(f"Loading dataset: {input_csv}")
df = pd.read_csv(input_csv)

# --- CHECK IF 'Traffic' COLUMN EXISTS --- #
if "Traffic" not in df.columns:
    raise ValueError("The dataset must have a 'Traffic' column.")

# --- GET UNIQUE TRAFFIC TYPES --- #
traffic_types = df["Traffic"].unique()
print("Found traffic types:", traffic_types)

# --- SEPARATE AND SAVE EACH TRAFFIC TYPE --- #
for t_type in traffic_types:
    filtered = df[df["Traffic"] == t_type]
    sanitized_name = t_type.replace(" ", "_")  # just in case
    output_path = os.path.join(output_dir, f"{sanitized_name}.csv")
    filtered.to_csv(output_path, index=False)
    print(f"Saved {len(filtered)} rows to {output_path}")
