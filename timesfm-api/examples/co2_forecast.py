"""Forecast CO2 levels using real Mauna Loa data."""

import requests
import pandas as pd


def download_co2_data():
    """Download CO2 data from NOAA."""
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.csv"
    
    # Read CSV, skip header rows
    df = pd.read_csv(url, comment='#', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    # Get the average column (weekly CO2 levels)
    co2_values = df['average'].values
    
    # Remove -999.99 (missing values)
    co2_values = co2_values[co2_values > 0]
    
    # Take last 365 points (about 7 years of weekly data)
    return co2_values[-365:].tolist()


def main():
    """Forecast CO2 levels."""
    print("Downloading CO2 data from NOAA Mauna Loa Observatory...")
    co2_data = download_co2_data()
    print(f"Loaded {len(co2_data)} weekly observations")
    print(f"Latest CO2 level: {co2_data[-1]:.2f} ppm")
    
    print("\nForecasting next 52 weeks...")
    response = requests.post(
        "http://localhost:8000/api/v1/forecast",
        json={"time_series": [co2_data], "horizon": 52},
        timeout=30.0
    )
    
    if response.status_code == 200:
        result = response.json()
        forecast = result['forecasts'][0]
        
        print(f"\nForecast (first 12 weeks):")
        for i in range(12):
            point = forecast['point_forecast'][i]
            q10 = forecast['quantiles']['q10'][i]
            q90 = forecast['quantiles']['q90'][i]
            print(f"  Week {i+1}: {point:.2f} ppm (80% interval: {q10:.2f} - {q90:.2f})")
        
        print(f"\nInference time: {result['metadata']['inference_time_ms']:.0f}ms")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
