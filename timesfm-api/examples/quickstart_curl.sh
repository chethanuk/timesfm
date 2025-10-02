#!/bin/bash

# TimesFM API Quickstart Examples with cURL
# This script demonstrates various ways to use the TimesFM API for time series forecasting

API_BASE_URL="http://localhost:8000"

echo "🚀 TimesFM API Quickstart with cURL"
echo "=================================="

# Function to check API health
check_health() {
    echo "1. Checking API health..."
    response=$(curl -s "$API_BASE_URL/health")
    if [ $? -eq 0 ]; then
        echo "✅ API is healthy"
        echo "$response" | jq '.'
    else
        echo "❌ API is not responding"
        exit 1
    fi
    echo ""
}

# Function to get model info
get_model_info() {
    echo "2. Getting model information..."
    curl -s "$API_BASE_URL/model/info" | jq '.'
    echo ""
}

# Function to make a basic forecast
basic_forecast() {
    echo "3. Basic forecast example..."
    curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{
            "data": [100, 102, 99, 103, 105, 108, 106, 109, 111, 108, 112, 115, 113, 116, 118, 120],
            "horizon": 8,
            "quantiles": [0.1, 0.5, 0.9]
        }' | jq '.'
    echo ""
}

# Function to forecast with stock-like data
stock_forecast() {
    echo "4. Stock price forecasting example..."
    curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{
            "data": [150.25, 152.30, 148.90, 155.40, 153.20, 158.60, 156.10, 160.80, 162.30, 159.70, 163.40, 165.80, 161.50, 167.20, 169.60, 166.80, 170.90, 172.40, 168.70, 174.30],
            "horizon": 10,
            "quantiles": [0.05, 0.5, 0.95]
        }' | jq '.'
    echo ""
}

# Function to forecast with seasonal data
seasonal_forecast() {
    echo "5. Seasonal pattern forecasting example..."
    curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{
            "data": [120, 135, 125, 140, 130, 145, 135, 150, 140, 155, 145, 160, 150, 165, 155, 170, 160, 175, 165, 180, 170, 185, 175, 190],
            "horizon": 12,
            "quantiles": [0.25, 0.5, 0.75]
        }' | jq '.'
    echo ""
}

# Function to demonstrate caching
demonstrate_caching() {
    echo "6. Demonstrating caching behavior..."

    # First request
    echo "Making first request..."
    start_time=$(date +%s%N)
    response1=$(curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{
            "data": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            "horizon": 5,
            "quantiles": [0.5]
        }')
    end_time=$(date +%s%N)
    first_time=$((($end_time - $start_time) / 1000000))

    echo "First request time: ${first_time}ms"
    echo "Cached: $(echo "$response1" | jq '.metadata.cached')"

    # Second request (should be cached)
    echo "Making second request (same data)..."
    start_time=$(date +%s%N)
    response2=$(curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{
            "data": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            "horizon": 5,
            "quantiles": [0.5]
        }')
    end_time=$(date +%s%N)
    second_time=$((($end_time - $start_time) / 1000000))

    echo "Second request time: ${second_time}ms"
    echo "Cached: $(echo "$response2" | jq '.metadata.cached')"
    echo ""
}

# Function to show different horizon lengths
horizon_examples() {
    echo "7. Different horizon examples..."

    # Short horizon
    echo "Short horizon (3 steps):"
    curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{
            "data": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "horizon": 3,
            "quantiles": [0.5]
        }' | jq '.metadata.horizon, .forecasts[:3]'

    # Medium horizon
    echo "Medium horizon (16 steps):"
    curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{
            "data": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "horizon": 16,
            "quantiles": [0.5]
        }' | jq '.metadata.horizon, .forecasts[:5]'

    echo ""
}

# Function to error handling examples
error_handling() {
    echo "8. Error handling examples..."

    # Empty data
    echo "Empty data request:"
    curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{"data": [], "horizon": 5, "quantiles": [0.5]}' | jq '.detail'

    # Invalid horizon
    echo "Negative horizon request:"
    curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d '{"data": [1, 2, 3], "horizon": -1, "quantiles": [0.5]}' | jq '.detail'

    echo ""
}

# Function to show performance metrics
performance_test() {
    echo "9. Performance test with larger dataset..."

    # Generate larger dataset
    large_data=$(python3 -c "
import json
data = [100 + i * 0.5 + (i % 10) * 2 for i in range(200)]
print(json.dumps(data))
")

    echo "Making forecast with 200 data points..."
    start_time=$(date +%s%N)
    response=$(curl -s -X POST "$API_BASE_URL/forecast" \
        -H "Content-Type: application/json" \
        -d "{
            \"data\": $large_data,
            \"horizon\": 32,
            \"quantiles\": [0.1, 0.5, 0.9]
        }")
    end_time=$(date +%s%N)
    total_time=$((($end_time - $start_time) / 1000000))

    echo "Total request time: ${total_time}ms"
    echo "Inference time: $(echo "$response" | jq '.metadata.inference_time')"
    echo "Input length: $(echo "$response" | jq '.metadata.input_length')"
    echo ""
}

# Function to show API statistics
show_stats() {
    echo "10. API statistics..."
    curl -s "$API_BASE_URL/stats" | jq '.'
    echo ""
}

# Main execution
main() {
    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        echo "⚠️  jq is required for formatted output. Please install jq."
        echo "On Ubuntu/Debian: sudo apt-get install jq"
        echo "On macOS: brew install jq"
        exit 1
    fi

    # Check if API is running
    echo "Checking if TimesFM API is running at $API_BASE_URL..."
    if ! curl -s "$API_BASE_URL/health" > /dev/null; then
        echo "❌ TimesFM API is not running at $API_BASE_URL"
        echo "Please start the API first:"
        echo "  cd /workspace/timesfm/timesfm-api"
        echo "  python3 app.py"
        exit 1
    fi

    echo "✅ TimesFM API is running!"
    echo ""

    # Run all examples
    check_health
    get_model_info
    basic_forecast
    stock_forecast
    seasonal_forecast
    demonstrate_caching
    horizon_examples
    error_handling
    performance_test
    show_stats

    echo "🎉 All examples completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Try with your own time series data"
    echo "2. Experiment with different horizon lengths"
    echo "3. Test various quantile combinations"
    echo "4. Use the API in your applications"
}

# Run examples if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi