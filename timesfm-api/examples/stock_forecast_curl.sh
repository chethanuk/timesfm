#!/bin/bash

# Stock Market Forecasting Example using cURL and TimesFM API
# This script demonstrates batch forecasting with multiple stocks, risk metrics, and portfolio allocation

set -e

# Configuration
API_BASE_URL="http://localhost:8000"  # Update with your TimesFM API endpoint
DATA_DIR="/workspace/timesfm/timesfm-api/data/stocks"
RESULTS_DIR="/workspace/timesfm/timesfm-api/curl_results"
LOG_FILE="$RESULTS_DIR/forecast.log"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check API health
check_api_health() {
    log "Checking API health..."

    if curl -s -f "$API_BASE_URL/health" > /dev/null; then
        log "API is healthy and responding"
        return 0
    else
        log "WARNING: API is not responding at $API_BASE_URL"
        log "This script assumes a TimesFM API server is running"
        log "Starting with mock data examples..."
        return 1
    fi
}

# Function to load stock data from JSON files
load_stock_data() {
    local ticker="$1"
    local data_file="$DATA_DIR/${ticker,,}_close.json"

    if [[ ! -f "$data_file" ]]; then
        log "ERROR: Stock data file not found: $data_file"
        return 1
    fi

    # Extract close prices from JSON
    local close_prices=$(jq -r '.close_prices | @tsv' "$data_file")
    local dates=$(jq -r '.dates | @tsv' "$data_file")

    log "Loaded $(echo "$close_prices" | wc -w) data points for $ticker"

    # Return data as variables
    echo "$close_prices"
    echo "$dates"
}

# Function to prepare forecast request payload
prepare_forecast_request() {
    local ticker="$1"
    local context_length="${2:-256}"
    local horizon="${3:-30}"

    local data_file="$DATA_DIR/${ticker,,}_close.json"

    if [[ ! -f "$data_file" ]]; then
        log "ERROR: Data file not found for $ticker"
        return 1
    fi

    # Create JSON payload for API request
    local payload=$(jq -n \
        --argjson data "$(jq '.close_prices[-'$context_length':]' "$data_file")" \
        --argjson horizon "$horizon" \
        --arg freq "0" \
        '{
            "data": [$data],
            "freq": [$freq],
            "horizon": [$horizon]
        }')

    echo "$payload"
}

# Function to make forecast request
make_forecast_request() {
    local ticker="$1"
    local payload="$2"

    log "Making forecast request for $ticker..."

    # Make API request
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$API_BASE_URL/forecast" 2>/dev/null || echo "API_ERROR")

    if [[ "$response" == "API_ERROR" ]]; then
        log "API request failed for $ticker, using mock forecast"
        create_mock_forecast "$ticker"
        return 0
    fi

    # Save response
    echo "$response" > "$RESULTS_DIR/${ticker}_forecast_response.json"
    log "Forecast response saved for $ticker"

    # Extract forecasts
    local forecasts=$(echo "$response" | jq -r '.forecasts[0] // empty')
    if [[ -n "$forecasts" ]]; then
        echo "$forecasts" > "$RESULTS_DIR/${ticker}_forecasts.txt"
        log "Forecasts extracted for $ticker"
    else
        log "No forecasts found in response for $ticker"
        create_mock_forecast "$ticker"
    fi
}

# Function to create mock forecasts when API is not available
create_mock_forecast() {
    local ticker="$1"
    local horizon="${2:-30}"

    log "Creating mock forecast for $ticker..."

    # Get last price from data
    local data_file="$DATA_DIR/${ticker,,}_close.json"
    local last_price=$(jq -r '.close_prices[-1]' "$data_file")

    # Generate mock forecasts with realistic patterns
    local forecasts=""
    local current_price="$last_price"

    for i in $(seq 1 $horizon); do
        # Add some realistic movement (0.5% daily volatility)
        local change=$(echo "scale=2; $current_price * $(awk -v seed=$RANDOM 'BEGIN{srand(seed); print (rand()-0.5)*0.01}')")
        current_price=$(echo "scale=2; $current_price + $change" | bc)
        # Ensure positive price
        current_price=$(echo "if ($current_price < 1) 1 else $current_price" | bc)

        if [[ -n "$forecasts" ]]; then
            forecasts="$forecasts,$current_price"
        else
            forecasts="$current_price"
        fi
    done

    echo "$forecasts" > "$RESULTS_DIR/${ticker}_forecasts.txt"

    # Create mock response JSON
    jq -n \
        --arg ticker "$ticker" \
        --argjson forecasts "[$forecasts]" \
        --arg last_price "$last_price" \
        --arg mock "true" \
        '{
            "ticker": $ticker,
            "forecasts": $forecasts,
            "last_price": ($last_price | tonumber),
            "mock": $mock
        }' > "$RESULTS_DIR/${ticker}_forecast_response.json"

    log "Mock forecast created for $ticker"
}

# Function to calculate risk metrics
calculate_risk_metrics() {
    local ticker="$1"
    local data_file="$DATA_DIR/${ticker,,}_full.json"
    local forecast_file="$RESULTS_DIR/${ticker,,}_forecasts.txt"

    if [[ ! -f "$data_file" ]] || [[ ! -f "$forecast_file" ]]; then
        log "ERROR: Required files not found for risk calculation for $ticker"
        return 1
    fi

    log "Calculating risk metrics for $ticker..."

    # Extract historical returns
    python3 - << EOF
import json
import numpy as np
import sys

try:
    # Load data
    with open("$data_file", 'r') as f:
        data = json.load(f)

    with open("$forecast_file", 'r') as f:
        forecasts = [float(x.strip()) for x in f.read().split(',')]

    # Extract historical prices and calculate returns
    prices = [item['Close'] for item in data['data']]
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

    # Calculate historical risk metrics
    returns_array = np.array(returns)
    volatility = np.std(returns_array) * np.sqrt(252)  # Annualized volatility
    var_95 = np.percentile(returns_array, 5)  # 95% VaR
    sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
    max_drawdown = 0
    peak = prices[0]

    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_drawdown = max(max_drawdown, drawdown)

    # Calculate forecast-based metrics
    last_price = prices[-1]
    forecast_returns = [(f - last_price) / last_price for f in forecasts]
    expected_return = np.mean(forecast_returns)
    forecast_volatility = np.std(forecast_returns) * np.sqrt(252)

    # Risk assessment
    risk_score = 0
    if volatility > 0.3:
        risk_score += 2
    elif volatility > 0.2:
        risk_score += 1

    if var_95 < -0.05:
        risk_score += 2
    elif var_95 < -0.03:
        risk_score += 1

    if max_drawdown > 0.3:
        risk_score += 2
    elif max_drawdown > 0.2:
        risk_score += 1

    risk_level = "Low"
    if risk_score >= 5:
        risk_level = "High"
    elif risk_score >= 3:
        risk_level = "Medium"

    # Create risk metrics JSON
    risk_metrics = {
        "ticker": "$ticker",
        "historical_metrics": {
            "volatility": round(volatility, 4),
            "var_95": round(var_95, 4),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 4),
            "annual_return": round((prices[-1]/prices[0] - 1), 4)
        },
        "forecast_metrics": {
            "expected_return": round(expected_return, 4),
            "forecast_volatility": round(forecast_volatility, 4),
            "forecast_horizon": len(forecasts)
        },
        "risk_assessment": {
            "risk_score": risk_score,
            "risk_level": risk_level
        }
    }

    # Save risk metrics
    with open("$RESULTS_DIR/${ticker,,}_risk_metrics.json", 'w') as f:
        json.dump(risk_metrics, f, indent=2)

    print(f"Risk metrics calculated for $ticker:")
    print(f"  Volatility: {volatility:.2%}")
    print(f"  95% VaR: {var_95:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Risk Level: {risk_level}")

except Exception as e:
    print(f"Error calculating risk metrics: {e}")
    sys.exit(1)
EOF
}

# Function to generate portfolio allocation
generate_portfolio_allocation() {
    local tickers=("AAPL" "MSFT" "GOOGL")
    local allocation_method="${1:-equal_weight}"  # equal_weight, risk_parity, momentum

    log "Generating portfolio allocation using $allocation_method method..."

    python3 - << EOF
import json
import numpy as np
import os

tickers = ${tickers[@]+${tickers[@]}}
allocation_method = "$allocation_method"
results_dir = "$RESULTS_DIR"

# Load risk metrics for all tickers
risk_metrics = {}
for ticker in tickers:
    risk_file = os.path.join(results_dir, f"{ticker}_risk_metrics.json")
    if os.path.exists(risk_file):
        with open(risk_file, 'r') as f:
            risk_metrics[ticker] = json.load(f)

# Load forecasts for all tickers
forecasts = {}
for ticker in tickers:
    forecast_file = os.path.join(results_dir, f"{ticker}_forecasts.txt")
    if os.path.exists(forecast_file):
        with open(forecast_file, 'r') as f:
            forecasts[ticker] = [float(x.strip()) for x in f.read().split(',')]

# Calculate allocations based on method
allocations = {}

if allocation_method == "equal_weight":
    # Equal weight allocation
    weight = 1.0 / len(tickers)
    for ticker in tickers:
        allocations[ticker] = weight

elif allocation_method == "risk_parity":
    # Risk parity allocation (inverse volatility)
    volatilities = {}
    for ticker in tickers:
        if ticker in risk_metrics:
            volatilities[ticker] = risk_metrics[ticker]['historical_metrics']['volatility']
        else:
            volatilities[ticker] = 0.2  # Default assumption

    inv_vols = {k: 1/v for k, v in volatilities.items()}
    total_inv_vol = sum(inv_vols.values())

    for ticker in tickers:
        allocations[ticker] = inv_vols[ticker] / total_inv_vol

elif allocation_method == "momentum":
    # Momentum-based allocation based on expected returns
    expected_returns = {}
    for ticker in tickers:
        if ticker in risk_metrics:
            expected_returns[ticker] = risk_metrics[ticker]['forecast_metrics']['expected_return']
        else:
            expected_returns[ticker] = 0.01  # Default assumption

    # Only allocate to positive expected returns
    positive_returns = {k: max(0, v) for k, v in expected_returns.items()}
    total_positive = sum(positive_returns.values())

    if total_positive > 0:
        for ticker in tickers:
            allocations[ticker] = positive_returns[ticker] / total_positive
    else:
        # Fallback to equal weight
        weight = 1.0 / len(tickers)
        for ticker in tickers:
            allocations[ticker] = weight

# Calculate portfolio metrics
portfolio_expected_return = 0
portfolio_volatility = 0
weights_list = []

for ticker in tickers:
    if ticker in allocations and ticker in risk_metrics:
        weight = allocations[ticker]
        expected_return = risk_metrics[ticker]['forecast_metrics']['expected_return']
        volatility = risk_metrics[ticker]['historical_metrics']['volatility']

        portfolio_expected_return += weight * expected_return
        portfolio_volatility += (weight * volatility) ** 2
        weights_list.append(weight)

portfolio_volatility = np.sqrt(portfolio_volatility)

# Create portfolio allocation JSON
portfolio_allocation = {
    "method": allocation_method,
    "tickers": tickers,
    "allocations": allocations,
    "portfolio_metrics": {
        "expected_return": round(portfolio_expected_return, 4),
        "volatility": round(portfolio_volatility, 4),
        "sharpe_ratio": round(portfolio_expected_return / portfolio_volatility, 2) if portfolio_volatility > 0 else 0,
        "diversification_ratio": round(len([w for w in weights_list if w > 0.01]) / len(tickers), 2)
    },
    "generated_at": "$(date -Iseconds)"
}

# Save portfolio allocation
with open(os.path.join(results_dir, "portfolio_allocation.json"), 'w') as f:
    json.dump(portfolio_allocation, f, indent=2)

print(f"Portfolio Allocation ({allocation_method}):")
for ticker, weight in allocations.items():
    print(f"  {ticker}: {weight:.2%}")

print(f"Portfolio Expected Return: {portfolio_expected_return:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
print(f"Portfolio Sharpe Ratio: {portfolio_expected_return/portfolio_volatility:.2f}" if portfolio_volatility > 0 else "Portfolio Sharpe Ratio: N/A")
EOF
}

# Function to create comprehensive report
create_report() {
    local report_file="$RESULTS_DIR/forecast_report_$(date +%Y%m%d_%H%M%S).md"

    log "Creating comprehensive forecast report..."

    cat > "$report_file" << EOF
# Stock Market Forecast Report

**Generated:** $(date)
**Method:** TimesFM API with cURL
**Tickers:** AAPL, MSFT, GOOGL

## Executive Summary

This report contains stock price forecasts generated using TimesFM, along with risk metrics and portfolio allocation recommendations.

## Individual Stock Forecasts

EOF

    # Add individual stock sections
    for ticker in AAPL MSFT GOOGL; do
        if [[ -f "$RESULTS_DIR/${ticker}_risk_metrics.json" ]]; then
            echo -e "\n### $ticker Analysis\n" >> "$report_file"
            python3 - << EOF >> "$report_file"
import json
import os

ticker = "$ticker"
results_dir = "$RESULTS_DIR"

# Load risk metrics
risk_file = os.path.join(results_dir, f"{ticker}_risk_metrics.json")
if os.path.exists(risk_file):
    with open(risk_file, 'r') as f:
        metrics = json.load(f)

    print(f"**Risk Level:** {metrics['risk_assessment']['risk_level']}")
    print(f"**Historical Volatility:** {metrics['historical_metrics']['volatility']:.2%}")
    print(f"**Expected Return:** {metrics['forecast_metrics']['expected_return']:.2%}")
    print(f"**Sharpe Ratio:** {metrics['historical_metrics']['sharpe_ratio']:.2f}")
    print(f"**Max Drawdown:** {metrics['historical_metrics']['max_drawdown']:.2%}")
EOF
        fi
    done

    # Add portfolio allocation section
    if [[ -f "$RESULTS_DIR/portfolio_allocation.json" ]]; then
        echo -e "\n## Portfolio Allocation\n" >> "$report_file"
        python3 - << EOF >> "$report_file"
import json
import os

results_dir = "$RESULTS_DIR"
portfolio_file = os.path.join(results_dir, "portfolio_allocation.json")

if os.path.exists(portfolio_file):
    with open(portfolio_file, 'r') as f:
        portfolio = json.load(f)

    print(f"**Allocation Method:** {portfolio['method']}")
    print(f"**Portfolio Expected Return:** {portfolio['portfolio_metrics']['expected_return']:.2%}")
    print(f"**Portfolio Volatility:** {portfolio['portfolio_metrics']['volatility']:.2%}")
    print(f"**Portfolio Sharpe Ratio:** {portfolio['portfolio_metrics']['sharpe_ratio']:.2f}")
    print(f"**Diversification Ratio:** {portfolio['portfolio_metrics']['diversification_ratio']:.2%}")

    print("\n**Allocations:**")
    for ticker, weight in portfolio['allocations'].items():
        print(f"- {ticker}: {weight:.2%}")
EOF
    fi

    # Add methodology section
    cat >> "$report_file" << EOF

## Methodology

1. **Data Loading:** Historical stock data loaded from JSON files
2. **Forecast Generation:** TimesFM model used to generate 30-day price forecasts
3. **Risk Metrics:** Calculated using historical returns and forecast expectations
4. **Portfolio Allocation:** Multiple allocation methods considered:
   - Equal Weight: Simple 1/N allocation
   - Risk Parity: Inverse volatility weighting
   - Momentum: Expected return-based allocation

## Risk Disclaimer

This report is for educational purposes only. Stock market forecasting involves significant risks, and past performance is not indicative of future results. Always consult with financial professionals before making investment decisions.

## Technical Details

- **Forecast Horizon:** 30 trading days
- **Historical Data:** 2 years of daily price data
- **Risk Metrics:** 95% VaR, Sharpe ratio, Maximum drawdown
- **API Endpoint:** $API_BASE_URL

---
*Report generated by TimesFM Stock Forecasting Script*
EOF

    log "Forecast report created: $report_file"
}

# Main execution function
main() {
    log "Starting TimesFM Stock Forecasting with cURL"

    # Check API health
    api_available=false
    if check_api_health; then
        api_available=true
    fi

    # Process each stock
    tickers=("AAPL" "MSFT" "GOOGL")

    for ticker in "${tickers[@]}"; do
        log "\n=== Processing $ticker ==="

        # Prepare and make forecast request
        if $api_available; then
            payload=$(prepare_forecast_request "$ticker" 256 30)
            if [[ -n "$payload" ]]; then
                make_forecast_request "$ticker" "$payload"
            else
                create_mock_forecast "$ticker"
            fi
        else
            create_mock_forecast "$ticker"
        fi

        # Calculate risk metrics
        calculate_risk_metrics "$ticker"
    done

    # Generate portfolio allocations
    log "\n=== Portfolio Allocation ==="
    generate_portfolio_allocation "equal_weight"
    generate_portfolio_allocation "risk_parity"
    generate_portfolio_allocation "momentum"

    # Create comprehensive report
    create_report

    log "\n=== Forecasting Complete ==="
    log "All results saved to: $RESULTS_DIR"
    log "Report available at: $RESULTS_DIR/forecast_report_*.md"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi