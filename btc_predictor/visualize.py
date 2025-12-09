"""
Visualization tool for Bitcoin predictions
Shows past 30 minutes of actual candles + 60 minute prediction
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ccxt
from model import BitcoinPredictor
import pickle

def fetch_live_data(minutes=200):
    """Fetch recent Bitcoin data for visualization"""
    print("Fetching live Bitcoin data from Binance...")
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'

    candles = exchange.fetch_ohlcv(symbol, '1m', limit=minutes)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    print(f"Fetched {len(df)} candles")
    print(f"Current BTC price: ${df['close'].iloc[-1]:,.2f}")

    return df

def predict_60min_price(predictor, data):
    """Predict price 60 minutes ahead"""
    # Use prepare_data but we only need the last sequence
    X, _ = predictor.prepare_data(data)

    if len(X) == 0:
        raise ValueError("Not enough data to make prediction")

    # Predict using the most recent data
    pred_normalized = predictor.predict(X[-1:])

    # Create a dummy feature array to inverse transform
    # We need all 14 features but only care about close price (index 3)
    dummy_features = np.zeros((1, 14))
    dummy_features[0, 3] = pred_normalized[0][0]  # Set close price

    # Inverse transform to get actual price
    pred_actual = predictor.scaler.inverse_transform(dummy_features)[0][3]

    return pred_actual

def create_candlestick_chart(data, predicted_price, save_path='btc_prediction.html'):
    """
    Create beautiful candlestick chart
    Shows past 30 mins (actual) + predicted price point at 60 mins
    """
    # Get last 30 minutes of actual data
    actual_data = data.iloc[-30:]

    # Create figure
    fig = go.Figure()

    # Add candlestick chart for past 30 minutes
    fig.add_trace(go.Candlestick(
        x=actual_data.index,
        open=actual_data['open'],
        high=actual_data['high'],
        low=actual_data['low'],
        close=actual_data['close'],
        name='Actual Price',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350',
        increasing_fillcolor='#26A69A',
        decreasing_fillcolor='#EF5350'
    ))

    # Calculate prediction point timestamp (60 minutes from now)
    current_time = actual_data.index[-1]
    prediction_time = current_time + timedelta(minutes=60)
    current_price = actual_data['close'].iloc[-1]

    # Add line from current price to prediction
    fig.add_trace(go.Scatter(
        x=[current_time, prediction_time],
        y=[current_price, predicted_price],
        mode='lines+markers',
        name='60-Min Prediction',
        line=dict(color='#FFA726', width=3, dash='dash'),
        marker=dict(size=[8, 15], color=['#FFA726', '#FF6F00'])
    ))

    # Add shaded region for prediction
    change_pct = ((predicted_price - current_price) / current_price) * 100
    color = '#26A69A' if change_pct >= 0 else '#EF5350'

    fig.add_trace(go.Scatter(
        x=[current_time, prediction_time, prediction_time, current_time],
        y=[current_price, predicted_price, predicted_price * 0.99, current_price],
        fill='toself',
        fillcolor=color,
        opacity=0.15,
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Add vertical line at "Now"
    fig.add_vline(
        x=current_time,
        line_dash="dot",
        line_color="#FFD700",
        line_width=2,
        annotation_text="NOW",
        annotation_position="top"
    )

    # Add annotations
    fig.add_annotation(
        x=prediction_time,
        y=predicted_price,
        text=f"${predicted_price:,.2f}<br>{change_pct:+.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#FF6F00",
        ax=-60,
        ay=-40,
        bgcolor="#1E1E1E",
        bordercolor="#FF6F00",
        borderwidth=2,
        font=dict(size=14, color="white")
    )

    # Update layout
    title_text = (
        f"<b>Bitcoin Price Prediction</b><br>"
        f"<sub>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</sub><br>"
        f"<sub>Current: ${current_price:,.2f} | "
        f"60-Min Prediction: ${predicted_price:,.2f} ({change_pct:+.2f}%)</sub>"
    )

    fig.update_layout(
        title=title_text,
        yaxis_title='Price (USDT)',
        xaxis_title='Time',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=700,
        hovermode='x unified',
        font=dict(size=12),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        xaxis=dict(gridcolor='#1E1E1E'),
        yaxis=dict(gridcolor='#1E1E1E')
    )

    # Save to HTML file
    fig.write_html(save_path)
    print(f"\n{'='*60}")
    print(f"Chart saved to: {save_path}")
    print(f"Open this file in your browser to see the prediction!")
    print(f"{'='*60}\n")

    return fig

def main():
    print("="*60)
    print("Bitcoin Price Prediction Visualizer")
    print("="*60)

    # Load trained model
    print("\nLoading trained model...")
    predictor = BitcoinPredictor()

    try:
        predictor.load_model('btc_predictor_60min.h5')

        # Load scaler
        try:
            with open('scaler.pkl', 'rb') as f:
                predictor.scaler = pickle.load(f)
        except:
            print("Warning: Could not load scaler, predictions may be inaccurate")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nModel not found. Please train the model first:")
        print("  py train.py")
        return

    # Fetch live data
    try:
        live_data = fetch_live_data(minutes=200)
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return

    # Make prediction
    print("\nGenerating 60-minute prediction...")
    try:
        predicted_price = predict_60min_price(predictor, live_data)

        current_price = live_data['close'].iloc[-1]
        change = ((predicted_price - current_price) / current_price) * 100

        print(f"\nCurrent BTC price: ${current_price:,.2f}")
        print(f"Predicted price (60 min): ${predicted_price:,.2f}")
        print(f"Expected change: {change:+.2f}%")

        # Create visualization
        create_candlestick_chart(live_data, predicted_price, 'btc_prediction.html')

    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
