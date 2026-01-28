#07_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# Load processed test data
data_dir = "../data/processed"
X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

# Prepare decoder input on the fly
def prepare_decoder_input(y):
    decoder_input = np.zeros_like(y)
    decoder_input[:, 1:] = y[:, :-1]
    return decoder_input[..., np.newaxis]

decoder_input_test = prepare_decoder_input(y_test)

# arima_model = joblib.load("../models/arima_model.pkl")
# seq2seq_model = load_model("../models/seq2seq_lstm.h5", compile=False)
# attention_model = load_model("../models/seq2seq_attention.h5", compile=False)

# Example
# predictions_arima = arima_model.forecast(steps=len(y_test))
# predictions_seq2seq = seq2seq_model.predict([X_test, decoder_input_test])
# predictions_attention = attention_model.predict([X_test, decoder_input_test])

def evaluate_model(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
    return rmse, mae, mape

attention_model = model   # your trained Seq2Seq + Attention model
seq2seq_model = model     # optionally, if you want baseline Seq2Seq

# ARIMA predictions (skip if no model)
try:
    arima_preds = []
    for i in range(len(y_test)):
        pred = arima_model.forecast(steps=y_test.shape[1])
        arima_preds.append(pred)
    arima_preds = np.array(arima_preds)
except NameError:
    arima_preds = None
    print("ARIMA model not found, skipping ARIMA predictions")

# Seq2Seq predictions
try:
    seq2seq_preds = seq2seq_model.predict([X_test, decoder_input_test])
except NameError:
    seq2seq_preds = None
    print("Seq2Seq model not found, skipping Seq2Seq predictions")

# Seq2Seq + Attention predictions
try:
    attention_preds = attention_model.predict([X_test, decoder_input_test])
except NameError:
    attention_preds = None
    print("Attention model not found, skipping Attention predictions")

plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten()[:50], label="Actual", marker='o')

if attention_preds is not None:
    plt.plot(attention_preds.flatten()[:50], label="Seq2Seq + Attention", marker='x')

if seq2seq_preds is not None:
    plt.plot(seq2seq_preds.flatten()[:50], label="Seq2Seq LSTM", marker='^')

if arima_preds is not None:
    plt.plot(arima_preds.flatten()[:50], label="ARIMA", marker='s')

plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("Forecast Comparison - First 50 Steps")
plt.legend()
plt.show()