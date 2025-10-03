# TimesFM API

Production-ready FastAPI service for time series forecasting using Google Research's TimesFM 2.5 foundation model.

## Features

- ✅ Zero-shot time series forecasting
- ✅ Quantile predictions for uncertainty quantification
- ✅ Batch processing support
- ✅ Production-ready with proper error handling
- ✅ Request tracing with unique IDs
- ✅ Health checks for Kubernetes
- ✅ GPU acceleration support

## Quick Start

### 1. Install Dependencies

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run the Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m app.main
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/api/v1/model/info

# Forecast (example)
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]],
    "horizon": 12
  }'
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
timesfm-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic
│   ├── api/                 # API routes
│   └── middleware/          # Middleware
├── tests/                   # Tests
├── pyproject.toml           # Dependencies
└── README.md
```

## Configuration

Key environment variables:

- `MODEL_NAME`: HuggingFace model ID (default: `google/timesfm-2.5-200m-pytorch`)
- `MODEL_CACHE_DIR`: Where to cache the model (default: `./model_cache`)
- `DEVICE`: Compute device (default: `cuda` if available, else `cpu`)
- `DEFAULT_MAX_CONTEXT`: Maximum context length (default: `1024`)
- `DEFAULT_MAX_HORIZON`: Maximum forecast horizon (default: `256`)

See `.env.example` for all options.

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# E2E tests (requires model download)
pytest tests/e2e/ -m e2e
```

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .
```

## Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16GB
- Disk: 10GB

**Recommended** (with GPU):
- CPU: 8+ cores
- RAM: 32GB
- GPU: NVIDIA T4 or better
- Disk: 50GB SSD

## License

Apache 2.0

## Citation

If you use TimesFM in your research, please cite:

```bibtex
@article{timesfm2024,
  title={A decoder-only foundation model for time-series forecasting},
  author={Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  journal={ICML},
  year={2024}
}
```
