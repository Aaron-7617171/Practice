#!/usr/bin/env python3
"""
Stock Market ML Example

A basic example of how to use machine learning with stock data.
This script demonstrates a simple LSTM model for stock price prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import glob
import argparse

def load_stock_data(data_dir, ticker):
    """
    Load stock data for a specific ticker from CSV files
    
    Args:
        data_dir (str): Directory containing the data files
        ticker (str): Ticker symbol to load
        
    Returns:
        DataFrame: Stock data
    """
    # Find the most recent CSV file for this ticker
    files = glob.glob(os.path.join(data_dir, f"{ticker}_*.csv"))
    if not files:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Sort by modification time (most recent first)
    latest_file = max(files, key=os.path.getmtime)
    
    # Load data
    df = pd.read_csv(latest_file)
    
    # Convert date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def prepare_data(df, target_col='Close', window_size=60, train_size=0.8):
    """
    Prepare data for LSTM model
    
    Args:
        df (DataFrame): Stock data
        target_col (str): Column to predict
        window_size (int): Number of previous days to use for prediction
        train_size (float): Proportion of data to use for training
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    # Select target column and convert to numpy array
    data = df[target_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create windowed dataset
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into train and test sets
    split = int(train_size * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model(window_size):
    """
    Build a simple LSTM model for stock prediction
    
    Args:
        window_size (int): Input sequence length
        
    Returns:
        Sequential: Keras LSTM model
    """
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_and_evaluate(ticker, data_dir='data', window_size=60, epochs=20, batch_size=32):
    """
    Train and evaluate a stock prediction model for a specific ticker
    
    Args:
        ticker (str): Ticker symbol
        data_dir (str): Directory containing data files
        window_size (int): Number of previous days to use
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (model, mse, predictions, actual_values)
    """
    # Load data
    df = load_stock_data(data_dir, ticker)
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        df, target_col='Close', window_size=window_size
    )
    
    # Build model
    model = build_lstm_model(window_size)
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate MSE
    mse = mean_squared_error(y_test_actual, predictions)
    
    return model, mse, predictions, y_test_actual

def plot_results(ticker, predictions, actual, save_dir='results'):
    """
    Plot the predicted vs actual stock prices
    
    Args:
        ticker (str): Ticker symbol
        predictions (array): Predicted values
        actual (array): Actual values
        save_dir (str): Directory to save the plot
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(14, 7))
    plt.plot(actual, color='blue', label=f'Actual {ticker} Price')
    plt.plot(predictions, color='red', label=f'Predicted {ticker} Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{ticker}_prediction.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction with LSTM')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--window', type=int, default=60, help='Window size (days)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    print(f"Training model for {args.ticker}...")
    
    # Train and evaluate model
    model, mse, predictions, actual = train_and_evaluate(
        args.ticker, 
        data_dir=args.data_dir,
        window_size=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Plot results
    plot_results(args.ticker, predictions, actual)
    print(f"Results saved to results/{args.ticker}_prediction.png")

if __name__ == "__main__":
    main() 