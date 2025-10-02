#!/usr/bin/env python3
"""
Create realistic synthetic stock data for TimesFM forecasting examples.
This script generates synthetic AAPL stock data when real data download fails.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os


def create_synthetic_stock_data(ticker="AAPL", years=2):
    """
    Create realistic synthetic stock data.

    Args:
        ticker (str): Stock ticker symbol
        years (int): Number of years of data to generate

    Returns:
        pd.DataFrame: Synthetic stock data
    """
    print(f"Creating synthetic {ticker} stock data for {years} years...")

    # Set seed for reproducibility
    np.random.seed(42)

    # Create date range (trading days only)
    start_date = datetime.now() - timedelta(days=365 * years)
    end_date = datetime.now()

    # Generate business days and filter out weekends
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    dates = dates[dates.weekday < 5]  # Remove weekends

    # Initial realistic price for AAPL 2 years ago
    if ticker == "AAPL":
        initial_price = 150.0
        daily_drift = 0.0008  # Positive trend
        volatility = 0.02     # 2% daily volatility
    elif ticker == "MSFT":
        initial_price = 250.0
        daily_drift = 0.0006
        volatility = 0.018
    elif ticker == "GOOGL":
        initial_price = 120.0
        daily_drift = 0.0005
        volatility = 0.022
    else:
        initial_price = 100.0
        daily_drift = 0.0005
        volatility = 0.02

    # Generate realistic price movements with some trend
    n_days = len(dates)

    # Add some trend and seasonality
    trend = np.linspace(0, 0.3, n_days)  # Gradual upward trend
    seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # Annual seasonality

    # Random walk with drift and seasonality
    returns = np.random.normal(daily_drift, volatility, n_days) + trend/252 + seasonal/252

    # Add some volatility clustering
    volatility_regime = np.random.choice([0.5, 1.0, 1.5, 2.0], n_days, p=[0.1, 0.6, 0.2, 0.1])
    volatility_regime = pd.Series(volatility_regime).rolling(20, min_periods=1).mean().values
    returns *= volatility_regime

    prices = [initial_price]
    for i in range(1, n_days):
        price_change = prices[-1] * returns[i]
        new_price = prices[-1] + price_change
        new_price = max(new_price, 1.0)  # Ensure price doesn't go negative
        prices.append(new_price)

    prices = np.array(prices)

    # Generate OHLC data (realistic relationships)
    intraday_range = 0.01 + 0.02 * np.random.random(n_days)  # 1-3% intraday range

    high = prices * (1 + intraday_range * np.abs(np.random.random(n_days)))
    low = prices * (1 - intraday_range * np.abs(np.random.random(n_days)))

    # Open price is typically between previous close and new close
    open_prices = np.zeros_like(prices)
    open_prices[0] = initial_price
    for i in range(1, n_days):
        gap = np.random.normal(0, 0.005)  # Small gap from previous close
        open_prices[i] = prices[i-1] * (1 + gap)

    # Realistic volume patterns
    base_volume = {
        "AAPL": 50000000,
        "MSFT": 30000000,
        "GOOGL": 40000000
    }.get(ticker, 30000000)

    volume = base_volume * (0.5 + 1.5 * np.random.random(n_days))
    volume = volume.astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices.round(2),
        'High': high.round(2),
        'Low': low.round(2),
        'Close': prices.round(2),
        'Volume': volume,
        'Dividends': 0.0,
        'Stock Splits': 0.0
    }, index=dates)

    # Add derived columns
    data['Ticker'] = ticker
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Day_Count'] = range(len(data))

    # Clean data - forward fill then backward fill any remaining NaN values
    data = data.ffill().bfill()

    print(f"Generated {len(data)} trading days of data")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"Average daily return: {data['Returns'].mean():.4f}")
    print(f"Annual volatility: {data['Returns'].std() * np.sqrt(252):.4f}")

    return data


def save_stock_data(data, ticker, save_dir="/workspace/timesfm/timesfm-api/data/stocks"):
    """
    Save stock data in multiple formats.

    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        save_dir (str): Directory to save the data
    """
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, ticker.lower())

    # Save as CSV
    csv_path = f"{base_path}.csv"
    data.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")

    # Save close prices as JSON (for TimesFM)
    close_data = {
        'ticker': ticker,
        'start_date': data.index.min().isoformat(),
        'end_date': data.index.max().isoformat(),
        'data_points': len(data),
        'close_prices': data['Close'].round(2).tolist(),
        'dates': data.index.strftime('%Y-%m-%d').tolist()
    }

    json_path = f"{base_path}_close.json"
    with open(json_path, 'w') as f:
        json.dump(close_data, f, indent=2)
    print(f"Saved JSON (close prices): {json_path}")

    # Save full data as JSON
    data_for_json = data.reset_index()
    if 'Date' not in data_for_json.columns and data.index.name is None:
        data_for_json = data.reset_index()
        data_for_json = data_for_json.rename(columns={'index': 'Date'})
    data_for_json['Date'] = data_for_json['Date'].dt.strftime('%Y-%m-%d')

    full_data = {
        'ticker': ticker,
        'start_date': data.index.min().isoformat(),
        'end_date': data.index.max().isoformat(),
        'data_points': len(data),
        'columns': list(data.columns),
        'data': data_for_json.to_dict('records')
    }

    full_json_path = f"{base_path}_full.json"
    with open(full_json_path, 'w') as f:
        json.dump(full_data, f, indent=2)
    print(f"Saved JSON (full data): {full_json_path}")


def validate_synthetic_data(data):
    """
    Validate synthetic data quality.

    Args:
        data (pd.DataFrame): Synthetic stock data
    """
    print("\n=== Data Quality Validation ===")

    # Check for basic consistency
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")

    # Check OHLC relationships
    invalid_ohlc = (data['High'] < data['Low']).sum()
    print(f"Invalid OHLC relationships: {invalid_ohlc}")

    # Check for negative prices
    negative_prices = (data['Close'] <= 0).sum()
    print(f"Negative prices: {negative_prices}")

    # Check for extreme returns
    returns = data['Returns'].dropna()
    extreme_returns = (np.abs(returns) > 0.2).sum()  # > 20% daily move
    print(f"Extreme daily returns (>20%): {extreme_returns}")

    # Basic statistics
    print(f"\nPrice Statistics:")
    print(f"  Min Close: ${data['Close'].min():.2f}")
    print(f"  Max Close: ${data['Close'].max():.2f}")
    print(f"  Mean Close: ${data['Close'].mean():.2f}")
    print(f"  Final Close: ${data['Close'].iloc[-1]:.2f}")
    print(f"  Total Return: {(data['Close'].iloc[-1]/data['Close'].iloc[0] - 1)*100:.1f}%")

    print(f"\nReturn Statistics:")
    print(f"  Mean Daily Return: {returns.mean():.4f}")
    print(f"  Daily Volatility: {returns.std():.4f}")
    print(f"  Annual Volatility: {returns.std() * np.sqrt(252):.4f}")
    print(f"  Sharpe Ratio: {(returns.mean()/returns.std()) * np.sqrt(252):.2f}")

    return invalid_ohlc == 0 and negative_prices == 0 and extreme_returns < len(returns) * 0.01


def main():
    """Main function to create synthetic stock data."""
    print("=== Synthetic Stock Data Generation ===")

    # Create data for multiple stocks
    tickers = ["AAPL", "MSFT", "GOOGL"]

    for ticker in tickers:
        print(f"\nGenerating data for {ticker}...")
        data = create_synthetic_stock_data(ticker, years=2)

        if data is not None:
            # Validate data quality
            is_valid = validate_synthetic_data(data)
            print(f"Data quality validation: {'PASSED' if is_valid else 'FAILED'}")

            # Save data
            save_stock_data(data, ticker)

            # Show sample
            print(f"\nSample data for {ticker}:")
            print(data[['Open', 'High', 'Low', 'Close', 'Volume']].head())
            print(f"\nLast 5 days for {ticker}:")
            print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
        else:
            print(f"Failed to create data for {ticker}")

    print("\n=== Synthetic Data Generation Complete ===")


if __name__ == "__main__":
    main()