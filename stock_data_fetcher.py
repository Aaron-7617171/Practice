#!/usr/bin/env python3
"""
Stock Data Fetcher

This script downloads historical stock data using the Yahoo Finance API 
and saves it to CSV files for later use in machine learning models.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import argparse

def fetch_stock_data(ticker_symbols, period='5y', interval='1d', output_dir='data'):
    """
    Download historical stock data for the given tickers.
    
    Args:
        ticker_symbols (list): List of ticker symbols (e.g., 'AAPL', 'MSFT')
        period (str): Period to download data for (e.g., '1d', '5d', '1mo', '1y', '5y', 'max')
        interval (str): Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        output_dir (str): Directory to save the data
    
    Returns:
        dict: Dictionary of DataFrame objects with stock data
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = {}
    
    print(f"Downloading data for {len(ticker_symbols)} stocks...")
    
    for symbol in ticker_symbols:
        try:
            print(f"Fetching data for {symbol}...")
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            
            # Basic preprocessing
            df = df.dropna()
            
            # Save to CSV
            file_path = os.path.join(output_dir, f"{symbol}_{interval}_{period}.csv")
            df.to_csv(file_path)
            print(f"Data saved to {file_path}")
            
            # Store in dictionary
            data[symbol] = df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return data

def fetch_stock_features(ticker_symbols, output_dir='data'):
    """
    Fetch additional stock features like company info and financial data.
    
    Args:
        ticker_symbols (list): List of ticker symbols
        output_dir (str): Directory to save the data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for symbol in ticker_symbols:
        try:
            print(f"Fetching features for {symbol}...")
            stock = yf.Ticker(symbol)
            
            # Get company info
            info = stock.info
            info_df = pd.DataFrame([info])
            info_df.to_csv(os.path.join(output_dir, f"{symbol}_info.csv"), index=False)
            
            # Get financial data
            try:
                financials = stock.financials
                financials.to_csv(os.path.join(output_dir, f"{symbol}_financials.csv"))
            except:
                print(f"No financial data available for {symbol}")
                
            # Get balance sheet
            try:
                balance_sheet = stock.balance_sheet
                balance_sheet.to_csv(os.path.join(output_dir, f"{symbol}_balance_sheet.csv"))
            except:
                print(f"No balance sheet data available for {symbol}")
                
            # Get cash flow
            try:
                cash_flow = stock.cashflow
                cash_flow.to_csv(os.path.join(output_dir, f"{symbol}_cash_flow.csv"))
            except:
                print(f"No cash flow data available for {symbol}")
                
        except Exception as e:
            print(f"Error fetching features for {symbol}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Stock Data Fetcher')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'], 
                        help='List of ticker symbols to fetch data for')
    parser.add_argument('--period', default='5y', help='Period to download data for')
    parser.add_argument('--interval', default='1d', help='Data interval')
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--features', action='store_true', help='Fetch additional stock features')
    
    args = parser.parse_args()
    
    # Fetch historical price data
    data = fetch_stock_data(args.tickers, args.period, args.interval, args.output)
    
    # Optionally fetch additional features
    if args.features:
        fetch_stock_features(args.tickers, args.output)
    
    print("Data download complete!")
    print(f"Downloaded data for {len(data)} stocks")
    
    # Print a sample of the data
    if data:
        first_ticker = next(iter(data))
        print(f"\nSample data for {first_ticker}:")
        print(data[first_ticker].head())

if __name__ == "__main__":
    main() 