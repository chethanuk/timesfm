# TimesFM API Examples

Examples demonstrating the API usage.

## Prerequisites

Start the API server:
```bash
cd /home/chethan/dev/timesfm/timesfm-api
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Examples

### 1. Simple Forecast
Basic forecast with monthly data.

```bash
uv run python examples/simple_forecast.py
```

### 2. Batch Forecast
Forecast multiple series at once.

```bash
uv run python examples/batch_forecast.py
```

### 3. CO2 Forecast
Forecast atmospheric CO2 levels using real NOAA data.

**Requirements**: `pip install pandas`

```bash
uv run python examples/co2_forecast.py
```

### 4. CSV Forecast
Forecast from your own CSV file.

**Requirements**: `pip install pandas`

```bash
uv run python examples/csv_forecast.py path/to/your/data.csv 12
```

CSV format: last column should be the values to forecast.

## Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/api/v1/model/info

# Quick forecast (12 data points)
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [[1,2,3,4,5,6,7,8,9,10,11,12]],
    "horizon": 6
  }'
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Notes

- TimesFM automatically pads short series - no minimum length required
- First API startup downloads ~800MB model (takes 1-3 minutes)
- Quantile ordering may not be guaranteed unless `fix_quantile_crossing=True` in config
- Default config is optimized for best performance
