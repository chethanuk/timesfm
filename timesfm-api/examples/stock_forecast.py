#!/usr/bin/env python3
"""
Stock Market Forecasting Example using TimesFM
This script demonstrates real-world stock forecasting with trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timesfm
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StockForecaster:
    """
    A comprehensive stock forecaster using TimesFM with trading strategy implementation.
    """

    def __init__(self, model_path=None, backend="gpu"):
        """
        Initialize the StockForecaster.

        Args:
            model_path (str): Path to TimesFM model
            backend (str): Backend to use ('gpu' or 'cpu')
        """
        self.model = None
        self.model_path = model_path
        self.backend = backend
        self.data = None
        self.forecasts = None
        self.trading_signals = None
        self.portfolio_value = None

        print(f"Initializing StockForecaster with {backend} backend...")

    def load_timesfm_model(self):
        """Load the TimesFM model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend=self.backend,
                        per_core_batch_size=32,
                        num_layers=20,
                        hidden_dims=1280
                    ),
                    checkpoint=self.model_path
                )
            else:
                # Use default model
                self.model = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend=self.backend,
                        per_core_batch_size=32,
                    )
                )
            print("TimesFM model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading TimesFM model: {e}")
            print("Using mock forecasting for demonstration...")
            return False

    def load_stock_data(self, ticker: str, data_path: str = None):
        """
        Load stock data from file.

        Args:
            ticker (str): Stock ticker symbol
            data_path (str): Path to stock data file
        """
        if data_path is None:
            data_path = f"/workspace/timesfm/timesfm-api/data/stocks/{ticker.lower()}.csv"

        try:
            self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            self.ticker = ticker
            print(f"Loaded {len(self.data)} days of data for {ticker}")
            print(f"Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
            return True
        except Exception as e:
            print(f"Error loading stock data: {e}")
            return False

    def preprocess_for_timesfm(self, context_length: int = 512, forecast_length: int = 64):
        """
        Preprocess data for TimesFM forecasting.

        Args:
            context_length (int): Number of past data points to use
            forecast_length (int): Number of future points to forecast
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load stock data first.")

        # Use close prices for forecasting
        prices = self.data['Close'].values

        # Prepare data for TimesFM
        self.context_data = prices[-context_length:].copy()
        self.forecast_length = forecast_length

        print(f"Using {len(self.context_data)} data points for forecasting")
        print(f"Forecasting {forecast_length} future points")

    def forecast_with_timesfm(self):
        """Generate forecasts using TimesFM model."""
        if self.model is None:
            print("TimesFM model not loaded. Using mock forecast...")
            return self._mock_forecast()

        try:
            # Prepare data in the format TimesFM expects
            # TimesFM expects a 2D array with shape (batch_size, sequence_length)
            forecast_input = self.context_data.reshape(1, -1)

            # Generate forecast
            forecast_output = self.model.forecast(
                forecast_input,
                freq=[0],  # Daily frequency
                horizon=[self.forecast_length]
            )

            # Extract forecast values
            self.forecasts = forecast_output[0]

            print(f"Generated {len(self.forecasts)} forecast points")
            return True

        except Exception as e:
            print(f"Error in TimesFM forecasting: {e}")
            print("Falling back to mock forecast...")
            return self._mock_forecast()

    def _mock_forecast(self):
        """Generate mock forecasts for demonstration when TimesFM is not available."""
        print("Generating mock forecasts for demonstration...")

        # Simple momentum-based mock forecast
        last_price = self.context_data[-1]
        recent_trend = np.mean(np.diff(self.context_data[-20:]))

        # Generate forecasts with some randomness and trend
        forecasts = []
        current_price = last_price

        for i in range(self.forecast_length):
            # Add trend and noise
            noise = np.random.normal(0, current_price * 0.01)  # 1% daily volatility
            trend_component = recent_trend * (1 - i / self.forecast_length)  # Decaying trend
            current_price = current_price + trend_component + noise
            current_price = max(current_price, 1.0)  # Ensure positive price
            forecasts.append(current_price)

        self.forecasts = np.array(forecasts)
        print(f"Generated {len(self.forecasts)} mock forecast points")
        return True

    def generate_trading_signals(self, strategy: str = "momentum"):
        """
        Generate trading signals based on forecasts.

        Args:
            strategy (str): Trading strategy ('momentum', 'mean_reversion', 'trend_following')
        """
        if self.forecasts is None:
            raise ValueError("No forecasts available. Please run forecasting first.")

        print(f"Generating {strategy} trading signals...")

        # Calculate expected returns from forecasts
        last_price = self.context_data[-1]
        expected_returns = (self.forecasts - last_price) / last_price

        # Generate signals based on strategy
        if strategy == "momentum":
            # Buy if expected return > 2%, sell if < -2%
            self.trading_signals = np.where(expected_returns > 0.02, 1,  # Buy
                                           np.where(expected_returns < -0.02, -1, 0))  # Sell/Hold

        elif strategy == "mean_reversion":
            # Simple mean reversion based on recent prices
            recent_mean = np.mean(self.context_data[-20:])
            current_price = last_price
            deviation = (current_price - recent_mean) / recent_mean

            # Buy if price is below mean, sell if above mean
            self.trading_signals = np.where(deviation < -0.05, 1,  # Buy
                                           np.where(deviation > 0.05, -1, 0))  # Sell/Hold

        elif strategy == "trend_following":
            # Trend following based on forecast direction
            forecast_trend = np.diff(self.forecasts)
            self.trading_signals = np.where(forecast_trend > 0, 1, -1)  # Buy if uptrend, sell if downtrend

        print(f"Generated {len(self.trading_signals)} trading signals")
        print(f"Signal distribution: Buy: {np.sum(self.trading_signals == 1)}, "
              f"Sell: {np.sum(self.trading_signals == -1)}, Hold: {np.sum(self.trading_signals == 0)}")

    def backtest_strategy(self, initial_capital: float = 10000, strategy: str = "momentum"):
        """
        Backtest the trading strategy.

        Args:
            initial_capital (float): Initial capital for backtesting
            strategy (str): Trading strategy to backtest
        """
        if self.trading_signals is None:
            self.generate_trading_signals(strategy)

        print(f"Backtesting {strategy} strategy with ${initial_capital:,.2f} initial capital...")

        # Simulate trading based on signals
        capital = initial_capital
        position = 0  # Number of shares held
        portfolio_values = []
        trades = []

        for i, signal in enumerate(self.trading_signals):
            current_price = self.forecasts[i] if i < len(self.forecasts) else self.context_data[-1]

            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                shares_to_buy = capital // current_price
                if shares_to_buy > 0:
                    position = shares_to_buy
                    capital -= position * current_price
                    trades.append({
                        'day': i,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': position,
                        'capital': capital
                    })

            elif signal == -1 and position > 0:  # Sell signal
                capital += position * current_price
                trades.append({
                    'day': i,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'capital': capital
                })
                position = 0

            # Calculate portfolio value
            portfolio_value = capital + position * current_price
            portfolio_values.append(portfolio_value)

        # Final position liquidation
        if position > 0:
            final_price = self.forecasts[-1]
            capital += position * final_price
            trades.append({
                'day': len(self.trading_signals) - 1,
                'action': 'SELL',
                'price': final_price,
                'shares': position,
                'capital': capital
            })

        self.portfolio_value = portfolio_values
        self.trades = trades
        self.final_capital = capital

        # Calculate performance metrics
        total_return = (self.final_capital - initial_capital) / initial_capital
        win_trades = len([t for t in trades if t['action'] == 'SELL' and t['capital'] > initial_capital])
        total_trades = len([t for t in trades if t['action'] == 'SELL'])

        print(f"\n=== Backtest Results ===")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${self.final_capital:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {win_trades}")
        print(f"Win Rate: {win_trades/total_trades:.2%}" if total_trades > 0 else "Win Rate: N/A")

        return {
            'total_return': total_return,
            'final_capital': self.final_capital,
            'total_trades': total_trades,
            'win_rate': win_trades/total_trades if total_trades > 0 else 0,
            'portfolio_values': portfolio_values
        }

    def plot_results(self, save_path: str = None):
        """
        Plot forecasting and trading results.

        Args:
            save_path (str): Path to save the plot
        """
        if self.data is None or self.forecasts is None:
            raise ValueError("No data or forecasts available for plotting.")

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{self.ticker} Stock Forecast and Trading Strategy', fontsize=16)

        # Plot 1: Historical prices and forecasts
        ax1 = axes[0]
        historical_days = len(self.context_data)
        forecast_days = len(self.forecasts)

        # Plot historical data
        historical_dates = self.data.index[-historical_days:]
        ax1.plot(historical_dates, self.context_data, 'b-', label='Historical Prices', linewidth=2)

        # Plot forecasts
        last_date = historical_dates[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
        ax1.plot(forecast_dates, self.forecasts, 'r--', label='Forecast', linewidth=2)

        ax1.set_title(f'{self.ticker} Price Forecast')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Trading signals
        ax2 = axes[1]
        if self.trading_signals is not None:
            signal_colors = ['red' if s == -1 else 'green' if s == 1 else 'gray' for s in self.trading_signals]
            ax2.scatter(forecast_dates[:len(self.trading_signals)],
                       [1] * len(self.trading_signals),
                       c=signal_colors, s=50, alpha=0.7)
            ax2.set_title('Trading Signals')
            ax2.set_ylabel('Signal')
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Sell', 'Hold', 'Buy'])
            ax2.grid(True, alpha=0.3)

        # Plot 3: Portfolio performance
        ax3 = axes[2]
        if self.portfolio_value is not None:
            ax3.plot(forecast_dates[:len(self.portfolio_value)],
                    self.portfolio_value, 'g-', linewidth=2)
            ax3.set_title('Portfolio Value')
            ax3.set_ylabel('Portfolio Value ($)')
            ax3.set_xlabel('Date')
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_results(self, save_dir: str = "/workspace/timesfm/timesfm-api/results"):
        """
        Save forecasting and trading results.

        Args:
            save_dir (str): Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            'ticker': self.ticker,
            'timestamp': timestamp,
            'forecast_length': self.forecast_length,
            'forecasts': self.forecasts.tolist() if self.forecasts is not None else None,
            'trading_signals': self.trading_signals.tolist() if self.trading_signals is not None else None,
            'portfolio_values': self.portfolio_value if hasattr(self, 'portfolio_value') else None,
            'final_capital': getattr(self, 'final_capital', None),
            'trades': getattr(self, 'trades', None)
        }

        results_path = os.path.join(save_dir, f"{self.ticker}_forecast_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_path}")
        return results_path


def main():
    """Main function to run the stock forecasting example."""
    print("=== Stock Market Forecasting Example ===")
    print("Using TimesFM for real-world stock forecasting and trading strategies\n")

    # Initialize forecaster
    forecaster = StockForecaster(backend="cpu")  # Use CPU for compatibility

    # Load TimesFM model (fallback to mock if not available)
    model_loaded = forecaster.load_timesfm_model()

    # Load stock data
    ticker = "AAPL"
    if not forecaster.load_stock_data(ticker):
        print("Failed to load stock data. Exiting...")
        return

    # Preprocess data for forecasting
    forecaster.preprocess_for_timesfm(context_length=256, forecast_length=30)

    # Generate forecasts
    forecast_success = forecaster.forecast_with_timesfm()

    # Generate trading signals
    forecaster.generate_trading_signals(strategy="momentum")

    # Backtest strategy
    backtest_results = forecaster.backtest_strategy(initial_capital=10000, strategy="momentum")

    # Plot results
    try:
        forecaster.plot_results()
    except Exception as e:
        print(f"Error plotting results: {e}")

    # Save results
    forecaster.save_results()

    print("\n=== Stock Forecasting Example Complete ===")


if __name__ == "__main__":
    main()