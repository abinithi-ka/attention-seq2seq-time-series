# 02_preprocessing.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configuration
RAW_DATA_PATH = "../data/raw/synthetic_energy_data.csv"
PROCESSED_DIR = "../data/processed"

ENCODER_LENGTH = 60     # past days used as input
DECODER_LENGTH = 14     # days to forecast

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Create output directory
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load dataset
data = pd.read_csv(RAW_DATA_PATH)

# Drop date column (kept only for plotting later)
features = data.drop(columns=["date"]).values

# Scale data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Create sequences
def create_sequences(data_array, enc_len, dec_len):
    X, y = [], []

    for i in range(len(data_array) - enc_len - dec_len):
        X.append(data_array[i : i + enc_len, :])
        y.append(data_array[i + enc_len : i + enc_len + dec_len, -1])

    return np.array(X), np.array(y)

X, y = create_sequences(
    scaled_features,
    ENCODER_LENGTH,
    DECODER_LENGTH)

# Train / Validation / Test split
# (no shuffling)
total_samples = len(X)

train_end = int(total_samples * TRAIN_RATIO)
val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Save processed arrays
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)

np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)

np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print("Preprocessing completed successfully.")
print(f"Train samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
