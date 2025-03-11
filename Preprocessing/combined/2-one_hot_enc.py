import time
import pandas as pd

# Tracks execution time
start_time = time.time()

print("Performing One Hot Encoding...")

# Load CSV file
df = pd.read_csv("removed_features.csv")

# One Hot Encoding for column "Sport"
df["Sport_wellknown"] = ((df["Sport"] >= 0) & (df["Sport"] <= 1023)).astype(int)
df["Sport_registered"] = ((df["Sport"] >= 1024) & (df["Sport"] <= 49151)).astype(int)
df["Sport_dynamic"] = ((df["Sport"] >= 49152) & (df["Sport"] <= 65535)).astype(int)

# One Hot Encoding for column "Dport"
df["Dport_wellknown"] = ((df["Dport"] >= 0) & (df["Dport"] <= 1023)).astype(int)
df["Dport_registered"] = ((df["Dport"] >= 1024) & (df["Dport"] <= 49151)).astype(int)
df["Dport_dynamic"] = ((df["Dport"] >= 49152) & (df["Dport"] <= 65535)).astype(int)

# One Hot Encoding for column "Proto"
proto_values = [0, 1, 2, 6, 17, 58, 2054, 35020]
for proto in proto_values:
    df[f"Proto_{proto}"] = (df["Proto"] == proto).astype(int)

# Remove columns "Sport", "Dport", and "Proto"
df = df.drop(columns=["Sport", "Dport", "Proto"])

# Save the processed CSV file
df.to_csv("one_hot_enc.csv", index=False)

# Check results
print(f"Columns: {df.columns.tolist()}")


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")
