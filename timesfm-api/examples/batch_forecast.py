"""Batch forecast example - multiple series at once."""

import requests
import numpy as np


def main():
    """Forecast multiple series in one request."""
    # Generate 3 different time series
    series1 = [100 + i * 2 + 10 * np.sin(i * 0.5) for i in range(50)]
    series2 = [50 + i * 1.5 - 5 * np.cos(i * 0.3) for i in range(60)]
    series3 = [200 + i * 3 + 15 * np.sin(i * 0.8) for i in range(40)]
    
    print(f"Forecasting 3 series:")
    print(f"  Series 1: {len(series1)} points")
    print(f"  Series 2: {len(series2)} points")
    print(f"  Series 3: {len(series3)} points")
    
    response = requests.post(
        "http://localhost:8000/api/v1/forecast",
        json={
            "time_series": [series1, series2, series3],
            "horizon": 12
        },
        timeout=30.0
    )
    
    if response.status_code == 200:
        result = response.json()
        
        for i, forecast in enumerate(result['forecasts'], 1):
            print(f"\nSeries {i} forecast (first 5):")
            print(f"  {forecast['point_forecast'][:5]}")
        
        print(f"\nBatch inference: {result['metadata']['inference_time_ms']:.0f}ms")
        print(f"Batch size: {result['metadata']['batch_size']}")
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    main()
