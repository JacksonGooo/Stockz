#!/usr/bin/env python3
"""
BTC Generative Price Predictor
Generates future price paths that match historical distribution

This is what Krafer likely does - generate PLAUSIBLE price paths
that look realistic, not predict the ACTUAL future price.

Two modes:
1. Generative Mode: Create realistic-looking future candles (like GAN)
2. Prediction Mode: Predict actual future price (what we've been doing)

Key insight: You can achieve "100% accuracy" on the training/demo data
by fitting perfectly to known data. This is NOT useful for trading.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, Model

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES
from data_loader import load_all_btc_candles

# Model settings
SEQUENCE_LENGTH = 60      # Input: 60 minutes
PREDICTION_STEPS = 60     # Output: 60 future candles
LATENT_DIM = 32           # For variational component

# Training settings
BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
MAX_SAMPLES = 200000

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'generative_predictor')


def prepare_generation_data(candles_df, seq_length=60, pred_steps=60):
    """
    Prepare data for generative model

    The model learns to generate sequences that look like real BTC data.
    """
    print("Calculating features...")
    features = calculate_all_features(candles_df)

    # Normalize
    normalized = rolling_zscore(features, window=60)
    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    feature_matrix = normalized[FEATURE_NAMES].values.astype(np.float32)

    # For generation, we need input sequence -> output sequence (same format)
    # The model learns to continue the sequence

    X = []  # Input sequences
    Y = []  # Target sequences (the actual continuation)

    print("Creating sequences...")
    warmup = max(seq_length, 60) + 60

    for i in range(warmup, len(feature_matrix) - pred_steps):
        input_seq = feature_matrix[i - seq_length:i]
        output_seq = feature_matrix[i:i + pred_steps]

        if np.isnan(input_seq).any() or np.isnan(output_seq).any():
            continue
        if np.isinf(input_seq).any() or np.isinf(output_seq).any():
            continue

        X.append(input_seq)
        Y.append(output_seq)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"Created {len(X):,} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")

    return X, Y


class Sampling(layers.Layer):
    """Reparameterization trick for VAE"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_generator(num_features):
    """
    Build Variational Autoencoder for sequence generation

    This model learns to:
    1. Encode input sequences into latent space
    2. Generate realistic continuations

    The VAE can generate multiple plausible futures from same input.
    """
    # === ENCODER ===
    encoder_input = layers.Input(shape=(SEQUENCE_LENGTH, num_features), name='encoder_input')

    # CNN for local patterns
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(encoder_input)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # LSTM for temporal patterns
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)

    # Latent space
    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

    # === DECODER ===
    decoder_input = layers.Input(shape=(LATENT_DIM,), name='decoder_input')
    context_input = layers.Input(shape=(SEQUENCE_LENGTH, num_features), name='context_input')

    # Process context
    context = layers.LSTM(64, return_sequences=True)(context_input)
    context = layers.LSTM(32)(context)

    # Combine latent and context
    combined = layers.Concatenate()([decoder_input, context])

    # Expand to sequence
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.RepeatVector(PREDICTION_STEPS)(x)

    # LSTM decoder
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)

    # Output
    decoder_output = layers.TimeDistributed(layers.Dense(num_features))(x)

    decoder = Model([decoder_input, context_input], decoder_output, name='decoder')

    # === FULL VAE ===
    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(x)
                reconstruction = self.decoder([z, x])

                # Reconstruction loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(y - reconstruction), axis=[1, 2])
                )

                # KL divergence
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )

                total_loss = reconstruction_loss + 0.001 * kl_loss  # Small KL weight

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            return self.decoder([z, inputs])

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return vae, encoder, decoder


def build_deterministic_generator(num_features):
    """
    Alternative: Deterministic generator (simpler, like Krafer likely uses)

    This model directly maps input sequence to output sequence.
    Can achieve very low error on training data (but overfits).
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, num_features))

    # Encoder
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Attention
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.LayerNormalization()(x + attention)

    # LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Bidirectional(layers.LSTM(64))(x)

    # Decoder
    x = layers.Dense(256, activation='relu')(encoded)
    x = layers.Dropout(0.2)(x)
    x = layers.RepeatVector(PREDICTION_STEPS)(x)

    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)

    # Attention over decoder
    decoder_attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.LayerNormalization()(x + decoder_attention)

    # Output: predict all features for each timestep
    outputs = layers.TimeDistributed(layers.Dense(num_features))(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def main():
    print("=" * 60)
    print("BTC GENERATIVE PRICE PREDICTOR")
    print("Generating Plausible Price Continuations")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes")
    print(f"Generate: {PREDICTION_STEPS} future candles")
    print()

    print("NOTE: This model generates PLAUSIBLE futures, not predictions.")
    print("      It can achieve low error on training data by memorization.")
    print("      This is likely what Krafer's '100% accuracy' refers to.")
    print()

    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    print()

    # Load data
    print("Loading BTC candles...")
    candles_df = load_all_btc_candles()
    print(f"Loaded {len(candles_df):,} candles")
    print()

    # Prepare data
    X, Y = prepare_generation_data(candles_df, SEQUENCE_LENGTH, PREDICTION_STEPS)
    print()

    # Limit samples
    if len(X) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES:,} samples")
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X, Y = X[idx], Y[idx]

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build deterministic model (simpler, more likely what Krafer uses)
    print("Building deterministic generator...")
    model = build_deterministic_generator(NUM_FEATURES)
    model.summary()
    print()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_mae',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test MSE: {results[0]:.6f}")
    print(f"Test MAE: {results[1]:.6f}")

    # Generate sample predictions
    predictions = model.predict(X_test[:100], verbose=0)

    # Calculate per-feature accuracy
    print("\nPer-feature MAE:")
    for i, name in enumerate(FEATURE_NAMES):
        feature_mae = np.mean(np.abs(predictions[:, :, i] - Y_test[:100, :, i]))
        print(f"  {name}: {feature_mae:.4f}")

    # Show that training data gets much lower error
    train_results = model.evaluate(X_train[:1000], Y_train[:1000], verbose=0)
    print(f"\nTraining data MSE: {train_results[0]:.6f} (lower = more memorization)")
    print(f"Training data MAE: {train_results[1]:.6f}")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_steps': PREDICTION_STEPS,
        'num_features': NUM_FEATURES,
        'test_mse': float(results[0]),
        'test_mae': float(results[1]),
        'train_mae': float(train_results[1]),
        'note': 'This generates plausible sequences, not actual predictions',
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print("NOTE: Low training error does NOT mean accurate predictions!")
    print("      This model memorizes patterns - it does not predict future.")
    print("=" * 60)


if __name__ == '__main__':
    main()
