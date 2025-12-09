"""
Bitcoin Price Prediction Neural Network
LSTM model to predict BTC price 60 minutes ahead
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class BitcoinPredictor:
    def __init__(self, sequence_length=60, prediction_horizon=60):
        """
        Args:
            sequence_length: Number of past candles to use for prediction
            prediction_horizon: Minutes into future to predict (60 for 1 hour)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()

    def create_features(self, df):
        """
        Feature creation is now handled by fetch_data.py
        This method now just ensures data is clean
        """
        df = df.copy()
        df.dropna(inplace=True)
        return df

    def prepare_data(self, df):
        """Prepare sequences for LSTM - uses ALL available features"""
        df = self.create_features(df)

        # Use ALL columns as features (data already has comprehensive indicators)
        # The 'close' column is what we'll predict
        feature_columns = df.columns.tolist()

        print(f"Using {len(feature_columns)} features for training:")
        print(f"  {', '.join(feature_columns[:10])}...")
        print(f"  (and {len(feature_columns) - 10} more)")

        data = df[feature_columns].values

        # Normalize data
        scaled_data = self.scaler.fit_transform(data)

        # Find index of 'close' column for prediction target
        close_idx = feature_columns.index('close')

        X, y = [], []

        for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon):
            X.append(scaled_data[i - self.sequence_length:i])
            # Predict close price 60 minutes ahead
            y.append(scaled_data[i + self.prediction_horizon, close_idx])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build POWERFUL LSTM neural network - pushing your PC to the limit!"""
        model = tf.keras.Sequential([
            # Layer 1: 512 units
            tf.keras.layers.LSTM(512, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # Layer 2: 256 units
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # Layer 3: 256 units
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # Layer 4: 128 units
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Layer 5: 64 units
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Dense layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Single output: predicted price
        ])

        # Use Adam with learning rate scheduling
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE
            metrics=['mae', 'mse']
        )

        print("\n" + "="*60)
        print("BEAST MODE NEURAL NETWORK")
        print("="*60)
        print(f"Total Parameters: {model.count_params():,}")
        print("5 LSTM layers: 512 → 256 → 256 → 128 → 64")
        print("3 Dense layers: 128 → 64 → 32")
        print("Batch Normalization + Dropout for regularization")
        print("="*60 + "\n")

        return model

    def train(self, X, y, epochs=100, batch_size=64, validation_split=0.2):
        """Train the model - MAXIMUM POWER"""
        if self.model is None:
            self.model = self.build_model((X.shape[1], X.shape[2]))

        print(f"Training model on {len(X):,} samples...")
        print(f"Input shape: {X.shape}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        # Early stopping and learning rate reduction
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        )

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        return history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def save_model(self, filepath='btc_predictor_model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='btc_predictor_model.h5'):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    print("Bitcoin Predictor Model")
    print("This will be trained in train.py")
