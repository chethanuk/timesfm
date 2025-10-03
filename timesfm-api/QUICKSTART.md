# TimesFM API - Quick Start

## Start API (One Command)

```bash
cd /home/chethan/dev/timesfm/timesfm-api
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First run downloads ~800MB model (1-3 minutes).

## Test It (Choose One)

### 1. Simple Test
```bash
uv run python examples/simple_forecast.py
```

### 2. Real CO2 Data
```bash
uv run python examples/co2_forecast.py
```

### 3. Your CSV File
```bash
uv run python examples/csv_forecast.py your_data.csv 12
```

### 4. cURL
```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"time_series": [[1,2,3,4,5,6,7,8,9,10,11,12]], "horizon": 6}'
```

## Docs

- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Model Info: http://localhost:8000/api/v1/model/info

## Key Facts

- ✅ No minimum length (TimesFM pads automatically)
- ✅ Works with 12, 24, 90, or any number of points
- ✅ Returns 10 quantiles (q00, q10, ..., q90)
- ✅ Point forecast = median (q50)
- ✅ Max input: 16,384 points
- ✅ Max horizon: 256 steps

## Files

- Examples: `examples/*.py` (4 clean scripts)
- Documentation: `FINAL_STATUS.md` (complete guide)
- All fixes: `ALL_FIXES_COMPLETE.md`

That's it! 🚀
