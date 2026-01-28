#04_baseline_seq2seq_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load preprocessed numpy arrays
data_dir = "../data/processed"

X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")

X_val = np.load(f"{data_dir}/X_val.npy")
y_val = np.load(f"{data_dir}/y_val.npy")

X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

# define model parameters
encoder_timesteps = X_train.shape[1]
num_features = X_train.shape[2]
decoder_timesteps = y_train.shape[1]

latent_dim = 64
learning_rate = 0.001

# encoder input
encoder_inputs = Input(shape=(encoder_timesteps, num_features))

# encoder LSTM
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder input
decoder_inputs = Input(shape=(decoder_timesteps, 1))

# decoder LSTM
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs,
    initial_state=encoder_states
)

# output layer
decoder_dense = Dense(1)
decoder_outputs = decoder_dense(decoder_outputs)

# build model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# compile model
model.compile(optimizer=Adam(learning_rate=learning_rate),loss="mse")
model.summary()

# prepare decoder input by shifting target
def prepare_decoder_input(y):
    decoder_input = np.zeros_like(y)
    decoder_input[:, 1:] = y[:, :-1]
    return decoder_input[..., np.newaxis]

decoder_input_train = prepare_decoder_input(y_train)
decoder_input_val = prepare_decoder_input(y_val)
decoder_input_test = prepare_decoder_input(y_test)

# train seq2seq model
history = model.fit(
    [X_train, decoder_input_train],
    y_train[..., np.newaxis],
    validation_data=(
        [X_val, decoder_input_val],
        y_val[..., np.newaxis]
    ),
    epochs=20,
    batch_size=32,
    verbose=1
)

# predict on test set
predictions = model.predict([X_test, decoder_input_test])

# flatten predictions and true values
y_true = y_test.flatten()
y_pred = predictions.flatten()

# calculate metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("Seq2Seq LSTM baseline evaluation")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MAPE: {mape:.2f}%")

"""This model uses an encoder decoder LSTM architecture to capture temporal dependencies across multiple input variables. However, it compresses all historical information into a single context vector, which limits its ability to focus on specific past time steps"""