import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Tracks execution time
start_time = time.time()

print("Performing Normalization...")

# Load CSV file
df = pd.read_csv("one_hot_enc.csv")

# Columns to be normalized
columns_to_normalize = ["SrcPkts", "DstPkts", "TotPkts", "DstBytes", "SrcBytes", "TotBytes",
                        "SrcLoad", "DstLoad", "Load", "SrcRate", "DstRate", "Rate", "SrcLoss",
                        "DstLoss", "Loss", "pLoss", "SrcJitter", "DstJitter", "SIntPkt", "DIntPkt",
                        "Dur", "TcpRtt", "IdleTime", "Sum", "Min", "Max", "sDSb", "SAppBytes", "DAppBytes",
                        "TotAppByte", "SynAck", "RunTime", "SrcJitAct", "DstJitAct",
                        "Mean", "sTtl", "dTtl", "sTos"]
# Features not included: sTtl, dTtl, sTos

# Initialize the MinMaxScaler (scales values to range 0-1)
scaler = MinMaxScaler(feature_range=(0,1))

# Apply Min-Max scaling to specified columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Save the normalized dataset
df.to_csv("normalized.csv", index=False, float_format="%.10f")


end_time = time.time()
print(f"\nTime it took to execute (in seconds): {end_time - start_time:.4f}")
