# 05_attention_seq2seq.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, TimeDistributed, Concatenate, RepeatVector
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

data_dir = "../data/processed"
X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")
X_val = np.load(f"{data_dir}/X_val.npy")
y_val = np.load(f"{data_dir}/y_val.npy")
X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

# Prepare decoder inputs by shifting target sequences
def prepare_decoder_input(y):
    decoder_input = np.zeros_like(y)
    decoder_input[:, 1:] = y[:, :-1]  # shift y by 1 timestep
    return decoder_input[..., np.newaxis]

decoder_input_train = prepare_decoder_input(y_train)
decoder_input_val = prepare_decoder_input(y_val)
decoder_input_test = prepare_decoder_input(y_test)

from tensorflow.keras.layers import Layer, Dense, Lambda
import tensorflow as tf

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, encoder_outputs, decoder_outputs):
        # encoder_outputs: (batch, enc_timesteps, latent_dim)
        # decoder_outputs: (batch, dec_timesteps, latent_dim)

        # Expand dims for broadcasting
        # decoder: (batch, dec_timesteps, 1, latent_dim)
        decoder_expanded = tf.expand_dims(decoder_outputs, axis=2)
        # encoder: (batch, 1, enc_timesteps, latent_dim)
        encoder_expanded = tf.expand_dims(encoder_outputs, axis=1)

        # Compute score: (batch, dec_timesteps, enc_timesteps, 1)
        score = tf.nn.tanh(self.W1(encoder_expanded) + self.W2(decoder_expanded))
        attention_weights = tf.nn.softmax(self.V(score), axis=2)

        # Compute context vector: sum over encoder timesteps
        context_vector = attention_weights * encoder_expanded
        context_vector = tf.reduce_sum(context_vector, axis=2)  # (batch, dec_timesteps, latent_dim)

        return context_vector, attention_weights

# Set model hyperparameters
encoder_timesteps = X_train.shape[1]
num_features = X_train.shape[2]
decoder_timesteps = y_train.shape[1]
latent_dim = 64
attention_units = 32
learning_rate = 0.001

# Build encoder
encoder_inputs = Input(shape=(encoder_timesteps, num_features))
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Build decoder
decoder_inputs = Input(shape=(decoder_timesteps, 1))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Apply Bahdanau Attention
attention_layer = BahdanauAttention(attention_units)
context_vector, attention_weights = attention_layer(encoder_outputs, decoder_outputs)

# Concatenate context vector with decoder outputs along last axis
decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])

# Apply TimeDistributed Dense to get final output
decoder_outputs_final = TimeDistributed(Dense(1))(decoder_combined)

# Build and compile the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
model.summary()

print("decoder_outputs shape:", decoder_outputs.shape)
print("context_vector shape:", context_vector.shape)

# Train the model
history = model.fit(
    [X_train, decoder_input_train],
    y_train[..., np.newaxis],
    validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Predict and evaluate
predictions = model.predict([X_test, decoder_input_test])
y_true = y_test.flatten()
y_pred = predictions.flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Seq2Seq + Attention evaluation")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MAPE: {mape:.2f}%")

# Visualize attention weights for first test sample
sample_encoder = X_test[:1]
sample_decoder = decoder_input_test[:1]

# Get encoder outputs
enc_outputs, enc_h, enc_c = encoder_lstm(sample_encoder)
decoder_state_h, decoder_state_c = enc_h, enc_c

# Get decoder outputs for sample
dec_outputs, _, _ = decoder_lstm(sample_decoder, initial_state=[decoder_state_h, decoder_state_c])

# Compute attention weights
context_vector, attention_weights = attention_layer(enc_outputs, dec_outputs)

# attention_weights shape: (batch, dec_timesteps, enc_timesteps, 1)
# Remove last singleton dimension for plotting
attention_matrix = attention_weights[0, :, :, 0].numpy()  # shape: (dec_timesteps, enc_timesteps)

plt.figure(figsize=(10, 6))
plt.imshow(attention_matrix, cmap="hot", aspect="auto")
plt.xlabel("Encoder timesteps")
plt.ylabel("Decoder timesteps")
plt.colorbar()
plt.title("Attention heatmap for first test sample")
plt.show()