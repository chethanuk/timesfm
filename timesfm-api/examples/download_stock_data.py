#!/usr/bin/env python3
"""
Download stock data from Yahoo Finance for TimesFM forecasting examples.
This script downloads AAPL stock data for the past 2 years and saves it.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json


def download_stock_data(ticker, period="2y", save_dir="/workspace/timesfm/timesfm-api/data/stocks"):
    """
    Download stock data for a given ticker and save it.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        save_dir (str): Directory to save the data
    """
    print(f"Downloading {ticker} data for period: {period}")

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Download stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            print(f"No data found for ticker {ticker}")
            return None

        print(f"Downloaded {len(data)} days of data for {ticker}")
        print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")

        # Clean and preprocess data
        data = preprocess_stock_data(data, ticker)

        # Save data in multiple formats
        save_stock_data(data, ticker, save_dir)

        return data

    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None


def preprocess_stock_data(data, ticker):
    """
    Clean and preprocess stock data for TimesFM.

    Args:
        data (pd.DataFrame): Raw stock data
        ticker (str): Stock ticker symbol

    Returns:
        pd.DataFrame: Preprocessed stock data
    """
    # Remove any rows with NaN values
    data = data.dropna()

    # Add some useful columns
    data['Ticker'] = ticker
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Add trading day counter
    data['Day_Count'] = range(len(data))

    # Forward fill any remaining NaN values from calculations
    data = data.fillna(method='ffill').fillna(method='bfill')

    print(f"Preprocessed data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    return data


def save_stock_data(data, ticker, save_dir):
    """
    Save stock data in multiple formats.

    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        save_dir (str): Directory to save the data
    """
    base_path = os.path.join(save_dir, ticker.lower())

    # Save as CSV
    csv_path = f"{base_path}.csv"
    data.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")

    # Save as JSON (close prices only for TimesFM)
    close_data = {
        'ticker': ticker,
        'start_date': data.index.min().isoformat(),
        'end_date': data.index.max().isoformat(),
        'data_points': len(data),
        'close_prices': data['Close'].tolist(),
        'dates': data.index.strftime('%Y-%m-%d').tolist()
    }

    json_path = f"{base_path}_close.json"
    with open(json_path, 'w') as f:
        json.dump(close_data, f, indent=2)
    print(f"Saved JSON (close prices): {json_path}")

    # Save as JSON (full data)
    full_data = {
        'ticker': ticker,
        'start_date': data.index.min().isoformat(),
        'end_date': data.index.max().isoformat(),
        'data_points': len(data),
        'columns': list(data.columns),
        'data': data.reset_index().to_dict('records')
    }

    full_json_path = f"{base_path}_full.json"
    with open(full_json_path, 'w') as f:
        json.dump(full_data, f, indent=2)
    print(f"Saved JSON (full data): {full_json_path}")


def validate_data_quality(data):
    """
    Validate the quality of downloaded stock data.

    Args:
        data (pd.DataFrame): Stock data to validate
    """
    print("\n=== Data Quality Validation ===")

    # Check for missing dates
    expected_days = len(data)
    actual_days = len(data.dropna())
    print(f"Expected days: {expected_days}, Actual days: {actual_days}")

    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found")

    # Check for outliers in returns
    returns = data['Returns'].dropna()
    q1, q3 = returns.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = returns[(returns < q1 - 1.5 * iqr) | (returns > q3 + 1.5 * iqr)]

    print(f"Found {len(outliers)} outlier returns ({len(outliers)/len(returns)*100:.1f}%)")

    # Basic statistics
    print(f"\nPrice Statistics:")
    print(f"  Min Close: ${data['Close'].min():.2f}")
    print(f"  Max Close: ${data['Close'].max():.2f}")
    print(f"  Mean Close: ${data['Close'].mean():.2f}")
    print(f"  Volatility (20d): {data['Volatility'].iloc[-1]:.4f}")

    return len(outliers) / len(returns) < 0.05  # Less than 5% outliers is acceptable


def main():
    """Main function to download stock data."""
    print("=== Stock Data Download Script ===")

    # Download AAPL data
    tickers = ['AAPL']

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        data = download_stock_data(ticker, period="2y")

        if data is not None:
            # Validate data quality
            is_valid = validate_data_quality(data)
            print(f"Data quality validation: {'PASSED' if is_valid else 'FAILED'}")

            # Show sample data
            print(f"\nSample data for {ticker}:")
            print(data.head())
            print(f"\nLast 5 days for {ticker}:")
            print(data.tail())
        else:
            print(f"Failed to download data for {ticker}")

    print("\n=== Download Complete ===")


if __name__ == "__main__":
    main()