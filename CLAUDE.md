# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses Poetry for dependency management. Key commands:

- **Installation**: `pip install -e .` (from project root)
- **Linting**: Uses ruff for code formatting and linting (configured in pyproject.toml)
- **Code style**: Line length 88, indent width 2

## Architecture Overview

TimesFM is a time series foundation model with the following key components:

### Main Package Structure
- `src/timesfm/` - Main package directory containing TimesFM 2.5 implementation
- `v1/` - Legacy TimesFM 1.0 and 2.0 implementations (archived)
- `pyproject.toml` - Poetry configuration and dependency management

### Core Architecture
- **`src/timesfm/__init__.py`** - Main API exports, exposes `TimesFM_2p5_200M_torch` and `ForecastConfig`
- **`src/timesfm/configs.py`** - Configuration classes including `ForecastConfig` with forecasting parameters
- **`src/timesfm/timesfm_2p5/`** - TimesFM 2.5 implementation
  - `timesfm_2p5_base.py` - Model definition and configuration
  - `timesfm_2p5_torch.py` - PyTorch implementation with `TimesFM_2p5_200M_torch_module`
- **`src/timesfm/torch/`** - Low-level PyTorch building blocks
  - `transformer.py` - Transformer layers
  - `dense.py` - Dense/residual blocks
  - `normalization.py` - Normalization utilities
  - `util.py` - General utilities

### Model Details
- **Current Version**: TimesFM 2.5 with 200M parameters
- **Architecture**: Decoder-only transformer model
- **Context Length**: Up to 16k timepoints
- **Output**: Point forecasts + optional quantile forecasts via 30M quantile head
- **Key Features**:
  - Patch-based input (input_patch_len=32, output_patch_len=128)
  - 20 transformer layers, 16 attention heads, 1280 model dimensions
  - Supports flip invariance, positivity constraints, quantile crossing fixes

### API Usage Pattern
1. Load model: `TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")`
2. Configure: `model.compile(ForecastConfig(...))`
3. Forecast: `model.forecast(horizon=N, inputs=[time_series_data])`

### Dependencies
- Python >=3.11
- PyTorch >=2.0.0 with CUDA
- NumPy >=1.26.4
- Hugging Face Hub >=0.23.0
- Safetensors >=0.5.3

### Legacy Support
The `v1/` directory contains TimesFM 1.0/2.0 implementations for backward compatibility. Use `pip install timesfm==1.3.0` for older versions.