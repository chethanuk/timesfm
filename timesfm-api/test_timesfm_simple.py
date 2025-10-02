#!/usr/bin/env python3
"""
Simple test script to debug TimesFM model usage
"""

import numpy as np
import torch
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

def test_timesfm():
    print("Testing TimesFM model...")

    # Set precision for better performance
    torch.set_float32_matmul_precision("high")

    # Load model
    print("Loading model...")
    model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    # Compile model
    print("Compiling model...")
    config = ForecastConfig(
        max_context=512,
        max_horizon=64,
        normalize_inputs=True,
        use_continuous_quantile_head=True
    )
    model.compile(config)

    # Test data
    test_data = [100, 102, 99, 103, 105, 108, 106, 109, 111, 108, 112, 115, 113, 116, 118, 120, 117, 121, 123, 125, 122, 126, 128, 130]

    print(f"Test data length: {len(test_data)}")

    # Test forecast
    print("Making forecast...")
    try:
        result = model.forecast(
            horizon=16,
            inputs=[np.array(test_data)]
        )
        print(f"Forecast successful!")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        if len(result) > 0:
            print(f"First result shape: {result[0].shape}")
        return True
    except Exception as e:
        print(f"Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_timesfm()
    if success:
        print("✓ TimesFM test passed")
    else:
        print("✗ TimesFM test failed")