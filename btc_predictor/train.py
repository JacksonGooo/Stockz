"""
Training script for Bitcoin price predictor
Loads data, trains model, evaluates performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import BitcoinPredictor
import pickle
import gzip
import os

def load_data(category='Crypto', asset='BTC'):
    """
    Load Bitcoin data from the master file with comprehensive features
    Uses unified Data/Category/Asset/ structure
    """
    from pathlib import Path

    # Try unified structure first
    master_file = Path(f'../Data/{category}/{asset}/{asset.lower()}_master.pkl.gz')

    # Fallback to old location
    if not master_file.exists():
        master_file = Path('data/btc_2year_master.pkl.gz')

    if master_file.exists():
        print(f"Loading comprehensive data from {master_file}...")
        with gzip.open(master_file, 'rb') as f:
            df = pickle.load(f)
        print(f"[OK] Loaded {len(df):,} candles with {len(df.columns)} features")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        return df
    else:
        raise FileNotFoundError(
            f"Data file not found! Run 'py fetch_data.py' first to process data."
        )

def evaluate_model(predictor, X_test, y_test):
    """Evaluate model performance"""
    predictions = predictor.predict(X_test)

    # Calculate metrics
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(mse)

    print("\n=== Model Performance ===")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    plt.plot(y_test[:500], label='Actual', alpha=0.7)
    plt.plot(predictions[:500], label='Predicted', alpha=0.7)
    plt.title('Bitcoin Price Predictions vs Actual (60 min ahead)')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig('predictions_plot.png')
    print("Plot saved to predictions_plot.png")

def main():
    # Load data (uses master file with comprehensive features!)
    df = load_data('Crypto', 'BTC')

    # Initialize predictor
    predictor = BitcoinPredictor(
        sequence_length=60,  # Use 60 past candles
        prediction_horizon=60  # Predict 60 minutes ahead
    )

    print("\nPreparing data...")
    X, y = predictor.prepare_data(df)

    print(f"\nData prepared:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series!
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model - BEAST MODE!
    print("\n=== Training Model - MAXIMUM POWER ===")
    history = predictor.train(
        X_train, y_train,
        epochs=100,  # Train longer for better accuracy
        batch_size=64,
        validation_split=0.2
    )

    # Save model and scaler
    predictor.save_model('btc_predictor_60min.h5')

    # Save scaler for later use in visualization
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(predictor.scaler, f)
    print("Scaler saved to scaler.pkl")

    # Evaluate
    print("\n=== Evaluating Model ===")
    evaluate_model(predictor, X_test, y_test)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")

if __name__ == "__main__":
    main()
