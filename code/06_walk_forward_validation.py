#06_walk_forward_validation.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, Concatenate, Layer
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
data_dir = "../data/processed"

X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")
X_val   = np.load(f"{data_dir}/X_val.npy")
y_val   = np.load(f"{data_dir}/y_val.npy")
X_test  = np.load(f"{data_dir}/X_test.npy")
y_test  = np.load(f"{data_dir}/y_test.npy")

print("Data loaded:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# Decoder input (teacher forcing)
def prepare_decoder_input(y):
    dec = np.zeros_like(y)
    dec[:, 1:] = y[:, :-1]
    return dec[..., np.newaxis]

decoder_input_train = prepare_decoder_input(y_train)
decoder_input_val   = prepare_decoder_input(y_val)
decoder_input_test  = prepare_decoder_input(y_test)


# Bahdanau Attention (Keras-safe)
class BahdanauAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V  = Dense(1)

    def call(self, encoder_outputs, decoder_outputs):
        enc_exp = tf.expand_dims(encoder_outputs, 1)
        dec_exp = tf.expand_dims(decoder_outputs, 2)

        score = tf.nn.tanh(self.W1(enc_exp) + self.W2(dec_exp))
        weights = tf.nn.softmax(self.V(score), axis=2)

        context = weights * enc_exp
        context = tf.reduce_sum(context, axis=2)

        return context, weights

#Model Builder
def build_seq2seq_attention_model(
    hidden_units=64,
    attention_units=32,
    learning_rate=0.0005,
    input_steps=60,
    output_steps=14,
    n_features=4
):
    encoder_inputs = Input(shape=(input_steps, n_features))
    enc_outputs, h, c = LSTM(
        hidden_units, return_sequences=True, return_state=True
    )(encoder_inputs)

    decoder_inputs = Input(shape=(output_steps, 1))
    dec_outputs = LSTM(
        hidden_units, return_sequences=True
    )(decoder_inputs, initial_state=[h, c])

    context, _ = BahdanauAttention(attention_units)(enc_outputs, dec_outputs)
    concat = Concatenate(axis=-1)([context, dec_outputs])

    outputs = TimeDistributed(Dense(1))(concat)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model
#BASELINE MODEL BUILDER (ADDED)

def build_baseline_seq2seq(
    hidden_units=64,
    learning_rate=0.0005,
    input_steps=60,
    output_steps=14,
    n_features=4
):
    encoder_inputs = Input(shape=(input_steps, n_features))
    _, h, c = LSTM(hidden_units, return_state=True)(encoder_inputs)

    decoder_inputs = Input(shape=(output_steps, 1))
    dec_outputs = LSTM(
        hidden_units, return_sequences=True
    )(decoder_inputs, initial_state=[h, c])

    outputs = TimeDistributed(Dense(1))(dec_outputs)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model

#Learning rate tuning
learning_rates = [1e-3, 5e-4, 1e-4]
lr_results = []

for lr in learning_rates:
    model = build_seq2seq_attention_model(learning_rate=lr)
    history = model.fit(
        [X_train, decoder_input_train],
        y_train[..., np.newaxis],
        validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
        epochs=5,
        batch_size=32,
        verbose=0
    )
    lr_results.append((lr, min(history.history["val_loss"])))

print("\nLearning rate tuning results:", lr_results)
best_lr = min(lr_results, key=lambda x: x[1])[0]

# Hidden units tuning
hidden_units_list = [32, 64, 128]
hu_results = []

for hu in hidden_units_list:
    model = build_seq2seq_attention_model(
        hidden_units=hu,
        learning_rate=best_lr
    )
    history = model.fit(
        [X_train, decoder_input_train],
        y_train[..., np.newaxis],
        validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
        epochs=5,
        batch_size=32,
        verbose=0
    )
    hu_results.append((hu, min(history.history["val_loss"])))

print("Hidden units tuning results:", hu_results)
best_hu = min(hu_results, key=lambda x: x[1])[0]

#TRAIN BASELINE MODEL
print("\nTraining Baseline Seq2Seq Model...")

baseline_model = build_baseline_seq2seq(
    hidden_units=best_hu,
    learning_rate=best_lr
)

baseline_model.fit(
    [X_train, decoder_input_train],
    y_train[..., np.newaxis],
    validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
    epochs=10,
    batch_size=32,
    verbose=1
)

#WALK-FORWARD (BASELINE)
print("\nRunning walk-forward validation (Baseline)...")

baseline_preds, baseline_true = [], []

for i in range(X_test.shape[0]):
    pred = baseline_model.predict(
        [X_test[i:i+1], decoder_input_test[i:i+1]],
        verbose=0
    )
    baseline_preds.append(pred[0, :, 0])
    baseline_true.append(y_test[i])

baseline_preds = np.array(baseline_preds)
baseline_true = np.array(baseline_true)

baseline_rmse = np.sqrt(mean_squared_error(
    baseline_true.flatten(), baseline_preds.flatten()
))
baseline_mae = mean_absolute_error(
    baseline_true.flatten(), baseline_preds.flatten()
)
baseline_mape = np.mean(
    np.abs((baseline_true - baseline_preds) / baseline_true)
) * 100

print("\nBaseline Results:")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"MAE: {baseline_mae:.4f}")
print(f"MAPE: {baseline_mape:.2f}%")

#Attention configuration tuning (EXPLICIT CONFIGS)
ATTENTION_CONFIGS = [
    {"type": "bahdanau", "units": 32},
    {"type": "bahdanau", "units": 64}
]

attn_results = []

for cfg in ATTENTION_CONFIGS:
    print(f"Testing attention config: {cfg}")

    model = build_seq2seq_attention_model(
        hidden_units=best_hu,
        attention_units=cfg["units"],
        learning_rate=best_lr
    )

    history = model.fit(
        [X_train, decoder_input_train],
        y_train[..., np.newaxis],
        validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
        epochs=5,
        batch_size=32,
        verbose=0
    )

    attn_results.append({
        "attention_units": cfg["units"],
        "val_loss": min(history.history["val_loss"])
    })

print("\nAttention configuration results:", attn_results)

best_attn = min(attn_results, key=lambda x: x["val_loss"])["attention_units"]
print("Selected attention units:", best_attn)


# Final model (selected config)
final_model = build_seq2seq_attention_model(
    hidden_units=best_hu,
    attention_units=best_attn,
    learning_rate=best_lr
)

final_model.fit(
    [X_train, decoder_input_train],
    y_train[..., np.newaxis],
    validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Walk-forward validation
print("\nRunning walk-forward validation...")

predictions, y_true = [], []

for i in range(X_test.shape[0]):
    pred = final_model.predict(
        [X_test[i:i+1], decoder_input_test[i:i+1]],
        verbose=0
    )
    predictions.append(pred[0, :, 0])
    y_true.append(y_test[i])

predictions = np.array(predictions)
y_true = np.array(y_true)

rmse = np.sqrt(mean_squared_error(y_true.flatten(), predictions.flatten()))
mae  = mean_absolute_error(y_true.flatten(), predictions.flatten())
mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100

print("\nWalk-forward results:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")


# Plots
for i in range(3):
    plt.figure(figsize=(8, 4))
    plt.plot(y_true[i], label="True")
    plt.plot(predictions[i], label="Predicted")
    plt.title(f"Walk-forward Sample {i+1}")
    plt.legend()
    plt.show()