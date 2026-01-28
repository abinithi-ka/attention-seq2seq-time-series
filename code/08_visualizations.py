# 08_visualizations.py
# Visualizations for Seq2Seq + Attention Forecasting

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model


# Load test data
data_dir = "../data/processed"

X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

decoder_input_test = np.zeros_like(y_test)
decoder_input_test[:, 1:] = y_test[:, :-1]
decoder_input_test = decoder_input_test[..., np.newaxis]

# Load trained models
# These must already exist in memory or be loaded
# attention_model = tf.keras.models.load_model("attention_model.h5", compile=False)
# seq2seq_model   = tf.keras.models.load_model("seq2seq_model.h5", compile=False)

# Generate predictions
attention_preds = attention_model.predict(
    [X_test, decoder_input_test], verbose=0
)
seq2seq_preds = seq2seq_model.predict(
    [X_test, decoder_input_test], verbose=0
)

attention_preds = attention_preds.squeeze()
seq2seq_preds = seq2seq_preds.squeeze()

# Forecast comparison plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten()[:50], label="Actual", marker="o")
plt.plot(attention_preds.flatten()[:50], label="Seq2Seq + Attention", marker="x")
plt.plot(seq2seq_preds.flatten()[:50], label="Seq2Seq", marker="^")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("Forecast Comparison (First 50 Steps)")
plt.legend()
plt.show()

# Attention heatmap (CORRECT WAY)
# Build a model that outputs attention weights
attention_layer = None
for layer in attention_model.layers:
    if "bahdanau" in layer.name.lower():
        attention_layer = layer
        break

if attention_layer is not None:
    attention_vis_model = Model(
        inputs=attention_model.inputs,
        outputs=attention_layer.output[1]  # attention weights
    )

    attn_weights = attention_vis_model.predict(
        [X_test[:1], decoder_input_test[:1]], verbose=0
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(attn_weights[0], aspect="auto", cmap="hot")
    plt.xlabel("Encoder Timesteps")
    plt.ylabel("Decoder Timesteps")
    plt.title("Attention Heatmap (Sample 1)")
    plt.colorbar()
    plt.show()
else:
    print("Attention layer not found – skipping heatmap")

# Error distribution
errors = attention_preds.flatten() - y_test.flatten()

rmse = np.sqrt(np.mean(errors ** 2))
mae = np.mean(np.abs(errors))
mape = np.mean(np.abs(errors / y_test.flatten())) * 100

plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title(
    f"Error Distribution (Seq2Seq + Attention)\n"
    f"RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%"
)
plt.show()