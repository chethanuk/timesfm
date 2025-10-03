"""Forecast from CSV file."""

import sys
import requests
import pandas as pd


def forecast_from_csv(csv_path: str, horizon: int = 12):
    """
    Read CSV and generate forecast.
    
    CSV should have columns: date, value
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Assume last column is the value
    values = df.iloc[:, -1].tolist()
    
    print(f"Loaded {len(values)} data points from {csv_path}")
    print(f"Last 5 values: {values[-5:]}")
    
    # Forecast
    print(f"\nForecasting {horizon} steps ahead...")
    response = requests.post(
        "http://localhost:8000/api/v1/forecast",
        json={"time_series": [values], "horizon": horizon},
        timeout=30.0
    )
    
    if response.status_code == 200:
        result = response.json()
        forecast = result['forecasts'][0]['point_forecast']
        
        print(f"\nForecast:")
        for i, value in enumerate(forecast, 1):
            print(f"  Step {i}: {value:.2f}")
        
        print(f"\nInference: {result['metadata']['inference_time_ms']:.0f}ms")
        return forecast
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python csv_forecast.py <path_to_csv> [horizon]")
        print("\nExample:")
        print("  python csv_forecast.py data.csv 12")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    
    forecast_from_csv(csv_path, horizon)


if __name__ == "__main__":
    main()
