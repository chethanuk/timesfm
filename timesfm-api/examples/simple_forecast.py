"""Simple forecast example with synthetic data."""

import requests


def main():
    """Simple forecast example."""
    # Monthly sales data (12 months)
    monthly_sales = [100, 120, 115, 130, 125, 140, 135, 150, 145, 160, 155, 170]
    
    # API request
    response = requests.post(
        "http://localhost:8000/api/v1/forecast",
        json={"time_series": [monthly_sales], "horizon": 6}
    )
    
    if response.status_code == 200:
        result = response.json()
        forecast = result['forecasts'][0]
        
        print("Forecast for next 6 months:")
        for i, value in enumerate(forecast['point_forecast'], 1):
            print(f"  Month {i}: {value:.2f}")
        
        print(f"\nInference time: {result['metadata']['inference_time_ms']:.2f}ms")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
