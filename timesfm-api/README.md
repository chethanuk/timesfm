# TimesFM API

A production-ready FastAPI application for time series forecasting using Google's TimesFM 2.5 model.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- PyTorch 2.0+
- 8GB+ RAM (for model loading)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/google-research/timesfm.git
   cd timesfm/timesfm-api
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   python3 -c "import timesfm; print('TimesFM installed successfully')"
   ```

### Starting the API

1. **Start the server:**
   ```bash
   python3 app.py
   ```

   The API will start on `http://localhost:8000`

2. **Start in background (optional):**
   ```bash
   nohup python3 app.py > logs/timesfm_api.log 2>&1 &
   ```

3. **Verify API health:**
   ```bash
   curl http://localhost:8000/health
   ```

### Quickstart with Real Datasets

Run the comprehensive examples script with real public datasets:

```bash
# Run all examples with real datasets
./examples/quickstart_public_datasets.sh

# Or set custom API base
API_BASE=http://localhost:8000 ./examples/quickstart_public_datasets.sh
```

**Featured Real Datasets:**
- **Sunspot Activity** (Monthly, 1749-2024) - Long-term seasonal patterns
- **Seasonal Patterns** - Monthly seasonal data with outliers
- **Linear Trends** - Growth and trend forecasting
- **Missing Value Data** - Real-world incomplete time series

**Example: Sunspot Forecast**
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [58.0, 62.6, 70.0, 55.7, 85.0, 83.5, 94.8, 66.3, 75.9, 71.6, 55.2, 54.8, 59.8, 62.5, 75.2, 85.7, 56.8, 54.3, 60.9, 61.4, 73.9, 77.7, 66.8, 66.9, 76.1, 69.8, 70.4, 85.9, 99.6, 85.2, 91.7, 98.2, 100.2, 95.8, 89.5, 68.1, 57.4, 58.8, 61.4, 68.1, 69.9, 62.6, 60.1, 79.4, 87.3, 94.2, 96.3, 96.1, 77.4],
    "horizon": 24,
    "quantiles": [0.1, 0.5, 0.9],
    "model_size": "200M"
  }'
```

## 📖 Usage Examples

### Data Preprocessing with Real Datasets

**Preprocess Sunspot Data (REVIN Normalization):**
```bash
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [58.0, 62.6, 70.0, 55.7, 85.0, 83.5, 94.8, 66.3, 75.9, 71.6, 55.2, 54.8, 59.8, 62.5, 75.2, 85.7],
    "method": "revin",
    "handle_missing": "interpolate",
    "remove_outliers": true,
    "outlier_threshold": 3.0,
    "smooth_data": false
  }'
```

**Handle Missing Values:**
```bash
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [10.2, null, 12.5, 13.8, null, 16.1, 17.4, 18.7, null, 21.0],
    "method": "zscore",
    "handle_missing": "interpolate",
    "remove_outliers": true,
    "outlier_threshold": 2.8
  }'
```

### cURL Examples

```bash
# Forecast with real sunspot data
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [58.0, 62.6, 70.0, 55.7, 85.0, 83.5, 94.8, 66.3, 75.9, 71.6, 55.2, 54.8, 59.8, 62.5, 75.2, 85.7, 56.8, 54.3, 60.9, 61.4, 73.9, 77.7, 66.8, 66.9, 76.1, 69.8, 70.4, 85.9, 99.6, 85.2, 91.7, 98.2, 100.2, 95.8, 89.5, 68.1, 57.4, 58.8, 61.4, 68.1, 69.9, 62.6, 60.1, 79.4, 87.3, 94.2, 96.3, 96.1, 77.4],
    "horizon": 24,
    "quantiles": [0.1, 0.5, 0.9],
    "model_size": "200M"
  }'

# Check available preprocessing methods
curl http://localhost:8000/preprocess/methods

# Check API health
curl http://localhost:8000/health
```

### Python Client Examples

```python
import requests
import json

# Load real sunspot data
with open('../../datasets/test_datasets.json') as f:
    datasets = json.load(f)
sunspot_data = datasets['sunspots_recent']

# Preprocess data
preprocess_response = requests.post("http://localhost:8000/preprocess", json={
    "data": sunspot_data,
    "method": "revin",
    "remove_outliers": True,
    "outlier_threshold": 3.0
})

# Forecast with preprocessed data
forecast_response = requests.post("http://localhost:8000/forecast", json={
    "data": sunspot_data,
    "horizon": 24,
    "quantiles": [0.1, 0.5, 0.9],
    "model_size": "200M"
})

result = forecast_response.json()
print(f"Request ID: {result['request_id']}")
print(f"Forecasts: {result['forecasts']}")
print(f"Quantiles: {result['quantiles']}")
print(f"Metadata: {result['metadata']}")
```

## 🔧 API Endpoints

### Data Preprocessing
```
POST /preprocess
```
Preprocess time series data using TimesFM 2.5 best practices.

**Parameters:**
- `data`: List[float] - Raw time series data
- `method`: str - Normalization method ("revin", "zscore", "minmax", "none")
- `handle_missing`: str - Missing value handling ("interpolate", "forward_fill", "backward_fill", "drop")
- `remove_outliers`: bool - Whether to remove outliers
- `outlier_threshold`: float - Z-score threshold for outlier detection
- `smooth_data`: bool - Whether to apply smoothing
- `window_size`: int - Window size for smoothing

### Available Preprocessing Methods
```
GET /preprocess/methods
```
Returns available preprocessing methods and best practices.

### Health Check
```
GET /health
```

Returns the API health status and model information.

### Model Information
```
GET /model/info
```

Returns detailed information about the loaded TimesFM model.

### Forecast
```
POST /forecast
```

Make time series forecasts.

**Request Body:**
```json
{
  "data": [100, 102, 99, 103, 105, 108, 106, 109, 111, 108, 112, 115],
  "horizon": 8,
  "quantiles": [0.1, 0.5, 0.9],
  "model_size": "200M"
}
```

**Parameters:**
- `data`: List of time series values (required)
- `horizon`: Number of future steps to predict (default: 48)
- `quantiles`: List of quantiles to predict (default: [0.1, 0.5, 0.9])
- `model_size`: Model size to use (default: "200M")

**Response:**
```json
{
  "request_id": "uuid",
  "forecasts": [120.5, 122.1, 124.3, ...],
  "quantiles": {
    "0.1": [115.2, 116.8, 118.9, ...],
    "0.5": [120.5, 122.1, 124.3, ...],
    "0.9": [125.8, 127.4, 129.6, ...]
  },
  "metadata": {
    "cached": false,
    "inference_time": 0.045,
    "model_size": "200M",
    "input_length": 12,
    "horizon": 8
  }
}
```

### Statistics
```
GET /stats
```

Returns API performance statistics and system information.

## 🧪 Testing

### Run E2E Tests

```bash
# Run all integration tests
python -m pytest tests/integration/test_timesfm_api_e2e.py -v

# Run specific test
python -m pytest tests/integration/test_timesfm_api_e2e.py::TestTimesFMEndpoint::test_health_endpoint_detailed -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Quick Test Script

```bash
# Test basic functionality
./examples/quickstart_curl.sh

# Run Python examples
python3 examples/python_client_example.py
```

## 📊 Model Details

### TimesFM 2.5 200M Model

- **Architecture**: Decoder-only transformer
- **Parameters**: 200M
- **Context Length**: Up to 512 timepoints (configurable)
- **Max Horizon**: Up to 64 steps (configurable)
- **Device**: CUDA GPU acceleration supported
- **Features**:
  - Zero-shot forecasting
  - Multiple quantile support
  - Input normalization
  - Confidence intervals
  - Caching for repeated requests

### Performance Characteristics

- **Typical inference time**: 20-50ms for most requests
- **Memory usage**: ~3-4GB GPU memory
- **Throughput**: 1000+ requests per second (with caching)
- **Caching**: Redis-based caching with 1-hour TTL

## ⚙️ Configuration

### Environment Variables

```bash
# Model Configuration
MODEL_SIZE=200M
MODEL_CHECKPOINT_PATH=/models/timesfm-200m.ckpt
PRELOAD_MODEL=true

# API Configuration
PORT=8000
HOST=0.0.0.0
WORKERS=1
LOG_LEVEL=INFO

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_TTL=3600

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

The model is compiled with these default settings:

```python
ForecastConfig(
    max_context=512,      # Maximum context length
    max_horizon=64,       # Maximum forecast horizon
    normalize_inputs=True,  # Enable input normalization
    use_continuous_quantile_head=True,  # Probabilistic forecasting
    force_flip_invariance=True,        # Mathematical properties
    infer_is_positive=True,            # Non-negative outputs
    fix_quantile_crossing=True        # Quantile consistency
)
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker/docker-compose.prod.yml logs -f timesfm-api

# Stop services
docker-compose -f docker/docker-compose.prod.yml down
```

### Individual Service

```bash
# Build Docker image
docker build -f docker/Dockerfile.prod -t timesfm-api .

# Run container
docker run -p 8000:8000 \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  timesfm-api
```

## 📈 Monitoring

### Health Monitoring

The API provides comprehensive health monitoring:

```bash
# Basic health
curl http://localhost:8000/health

# Detailed stats
curl http://localhost:8000/stats

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Monitoring Stack

The full monitoring stack includes:

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Redis**: Request caching and session storage
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics

## 🔒 Security

### API Security

- CORS middleware for cross-origin requests
- Request validation and sanitization
- Rate limiting (configurable)
- Input size limits
- Error handling without information leakage

### Production Deployment

For production deployment:

1. **Use HTTPS** with proper SSL certificates
2. **Configure firewall** rules
3. **Set up authentication** middleware
4. **Monitor logs** for suspicious activity
5. **Implement rate limiting** per client
6. **Use environment variables** for sensitive configuration

## 🚀 Performance Optimization

### Caching

- Redis-based caching for identical requests
- Cache key based on data hash and parameters
- TTL of 1 hour for cached results
- Automatic cache invalidation on model reload

### GPU Acceleration

- Automatic CUDA detection and usage
- Mixed precision training support
- Memory-efficient inference
- Batch processing capabilities

### Request Optimization

- Async request handling
- Connection pooling
- Gzip compression for responses
- Efficient JSON serialization

## 🛠️ Development

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/google-research/timesfm.git
cd timesfm/timesfm-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements.prod.txt

# Start development server
python3 app.py
```

### Code Style

- **Linter**: `ruff check src/`
- **Formatter**: `ruff format src/`
- **Type checking**: `mypy src/`
- **Testing**: `pytest tests/`

### Adding New Features

1. Add tests in `tests/integration/`
2. Update API documentation
3. Add examples in `examples/`
4. Update CHANGELOG.md
5. Test with different data patterns

## 📝 Examples

### Basic Usage

```python
import requests

# Simple forecast
response = requests.post("http://localhost:8000/forecast", json={
    "data": [100, 102, 99, 103, 105, 108, 106, 109],
    "horizon": 5,
    "quantiles": [0.1, 0.5, 0.9]
})

result = response.json()
print(f"Forecast: {result['forecasts']}")
```

### Batch Forecasting

```python
# Multiple forecasts
data_series = [
    [100, 102, 99, 103, 105],  # Series 1
    [200, 202, 198, 203, 205],  # Series 2
]

forecasts = []
for data in data_series:
    response = requests.post("http://localhost:8000/forecast", json={
        "data": data,
        "horizon": 10,
        "quantiles": [0.5]
    })
    forecasts.append(response.json()['forecasts'])

print(f"Batch forecasts: {forecasts}")
```

### Confidence Intervals

```python
# Get confidence intervals
response = requests.post("http://localhost:8000/forecast", json={
    "data": [100, 102, 99, 103, 105, 108, 106, 109],
    "horizon": 5,
    "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]
})

result = response.json()
for i in range(5):
    q05 = result['quantiles']['0.05'][i]
    q95 = result['quantiles']['0.95'][i]
    median = result['quantiles']['0.5'][i]
    print(f"Step {i+1}: [{q05:.2f}, {median:.2f}, {q95:.2f}]")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Research for the TimesFM model
- Hugging Face for model hosting
- FastAPI for the web framework
- The open-source community for tools and libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/google-research/timesfm/issues)
- **Documentation**: [TimesFM Research](https://github.com/google-research/timesfm)
- **Model Hub**: [Hugging Face](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

---

**⚡ Powered by Google's TimesFM 2.5 - State-of-the-art time series foundation model**