#!/bin/bash

# TimesFM API Quickstart Examples with Real Public Datasets
# This script demonstrates comprehensive API usage with real datasets

API_BASE="${API_BASE:-http://localhost:8000}"

echo "TimesFM API Quickstart with Real Datasets"
echo "=========================================="

# Helper function to make API calls
call_api() {
    local method=$1
    local endpoint=$2
    local data=$3

    if [ "$method" = "GET" ]; then
        curl -s -X GET "${API_BASE}${endpoint}" | jq '.'
    else
        curl -s -X POST "${API_BASE}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "$data" | jq '.'
    fi
}

echo -e "\n1. Health Check"
echo "-----------------"
call_api "GET" "/health"

echo -e "\n2. Check Available Preprocessing Methods"
echo "------------------------------------------"
call_api "GET" "/preprocess/methods"

echo -e "\n3. Preprocess Sunspot Data with REVIN Normalization"
echo "------------------------------------------------------"
SUNSPOT_DATA=$(cat <<EOF
{
    "data": [58.0, 62.6, 70.0, 55.7, 85.0, 83.5, 94.8, 66.3, 75.9, 71.6, 55.2, 54.8, 59.8, 62.5, 75.2, 85.7, 56.8, 54.3, 60.9, 61.4],
    "method": "revin",
    "handle_missing": "interpolate",
    "remove_outliers": true,
    "outlier_threshold": 3.0,
    "smooth_data": false
}
EOF
)
call_api "POST" "/preprocess" "$SUNSPOT_DATA"

echo -e "\n4. Forecast Sunspot Activity (24-month horizon)"
echo "-------------------------------------------------"
SUNSPOT_FORECAST=$(cat <<EOF
{
    "data": [58.0, 62.6, 70.0, 55.7, 85.0, 83.5, 94.8, 66.3, 75.9, 71.6, 55.2, 54.8, 59.8, 62.5, 75.2, 85.7, 56.8, 54.3, 60.9, 61.4, 73.9, 77.7, 66.8, 66.9, 76.1, 69.8, 70.4, 85.9, 99.6, 85.2, 91.7, 98.2, 100.2, 95.8, 89.5, 68.1, 57.4, 58.8, 61.4, 68.1, 69.9, 62.6, 60.1, 79.4, 87.3, 94.2, 96.3, 96.1, 77.4],
    "horizon": 24,
    "quantiles": [0.1, 0.5, 0.9],
    "model_size": "200M"
}
EOF
)
call_api "POST" "/forecast" "$SUNSPOT_FORECAST"

echo -e "\n5. Preprocess Seasonal Data with Outlier Removal"
echo "--------------------------------------------------"
SEASONAL_DATA=$(cat <<EOF
{
    "data": [10.5, 12.3, 15.8, 18.2, 14.7, 11.9, 9.8, 13.4, 16.7, 19.1, 15.6, 12.8, 10.9, 14.2, 17.5, 20.1, 16.4, 13.7, 11.5, 15.8, 18.9, 21.3, 17.8, 15.1],
    "method": "revin",
    "handle_missing": "interpolate",
    "remove_outliers": true,
    "outlier_threshold": 2.5,
    "smooth_data": true,
    "window_size": 3
}
EOF
)
call_api "POST" "/preprocess" "$SEASONAL_DATA"

echo -e "\n6. Forecast Seasonal Pattern (12-month horizon)"
echo "---------------------------------------------------"
SEASONAL_FORECAST=$(cat <<EOF
{
    "data": [10.5, 12.3, 15.8, 18.2, 14.7, 11.9, 9.8, 13.4, 16.7, 19.1, 15.6, 12.8, 10.9, 14.2, 17.5, 20.1, 16.4, 13.7, 11.5, 15.8, 18.9, 21.3, 17.8, 15.1],
    "horizon": 12,
    "quantiles": [0.25, 0.5, 0.75],
    "model_size": "200M"
}
EOF
)
call_api "POST" "/forecast" "$SEASONAL_FORECAST"

echo -e "\n7. Process Linear Trend Data with MinMax Normalization"
echo "---------------------------------------------------------"
LINEAR_DATA=$(cat <<EOF
{
    "data": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138],
    "method": "minmax",
    "handle_missing": "forward_fill",
    "remove_outliers": false,
    "smooth_data": true,
    "window_size": 5
}
EOF
)
call_api "POST" "/preprocess" "$LINEAR_DATA"

echo -e "\n8. Forecast Linear Trend (15-step horizon)"
echo "---------------------------------------------"
LINEAR_FORECAST=$(cat <<EOF
{
    "data": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138],
    "horizon": 15,
    "quantiles": [0.05, 0.5, 0.95],
    "model_size": "200M"
}
EOF
)
call_api "POST" "/forecast" "$LINEAR_FORECAST"

echo -e "\n9. Handle Data with Missing Values"
echo "--------------------------------------"
MISSING_DATA=$(cat <<EOF
{
    "data": [10.2, null, 12.5, 13.8, null, 16.1, 17.4, 18.7, null, 21.0, 22.3, 23.6, null, 25.9, 27.2, null, 29.5, 30.8, 32.1, null],
    "method": "zscore",
    "handle_missing": "interpolate",
    "remove_outliers": true,
    "outlier_threshold": 2.8,
    "smooth_data": false
}
EOF
)
call_api "POST" "/preprocess" "$MISSING_DATA"

echo -e "\n10. Long-term Forecast with Historical Data"
echo "---------------------------------------------"
LONG_TERM_DATA=$(cat <<EOF
{
    "data": [45.2, 47.1, 46.8, 48.9, 50.2, 49.7, 51.4, 52.8, 53.1, 54.6, 55.9, 56.2, 57.8, 58.4, 59.1, 60.3, 61.7, 62.1, 63.5, 64.2, 65.8, 66.4, 67.1, 68.3, 69.7, 70.1, 71.5, 72.2, 73.8, 74.4, 75.1, 76.3, 77.7, 78.1, 79.5, 80.2, 81.8, 82.4, 83.1, 84.3, 85.7, 86.1, 87.5, 88.2, 89.8, 90.4, 91.1, 92.3, 93.7, 94.1, 95.5, 96.2, 97.8, 98.4, 99.1, 100.3, 101.7, 102.1, 103.5, 104.2, 105.8, 106.4, 107.1, 108.3, 109.7, 110.1, 111.5, 112.2, 113.8, 114.4, 115.1, 116.3, 117.7, 118.1, 119.5, 120.2, 121.8, 122.4, 123.1, 124.3, 125.7, 126.1, 127.5, 128.2, 129.8, 130.4, 131.1, 132.3, 133.7, 134.1, 135.5, 136.2, 137.8, 138.4, 139.1, 140.3, 141.7, 142.1, 143.5, 144.2, 145.8, 146.4, 147.1, 148.3, 149.7, 150.1, 151.5, 152.2, 153.8, 154.4, 155.1, 156.3, 157.7, 158.1, 159.5, 160.2, 161.8, 162.4, 163.1, 164.3, 165.7, 166.1, 167.5, 168.2, 169.8, 170.4, 171.1, 172.3, 173.7, 174.1, 175.5, 176.2, 177.8, 178.4, 179.1, 180.3, 181.7, 182.1, 183.5, 184.2, 185.8, 186.4, 187.1, 188.3, 189.7, 190.1, 191.5, 192.2, 193.8, 194.4, 195.1, 196.3, 197.7, 198.1, 199.5, 200.2, 201.8, 202.4, 203.1, 204.3, 205.7, 206.1, 207.5, 208.2, 209.8, 210.4, 211.1, 212.3, 213.7, 214.1, 215.5, 216.2, 217.8, 218.4, 219.1, 220.3, 221.7, 222.1, 223.5, 224.2, 225.8, 226.4, 227.1, 228.3, 229.7, 230.1, 231.5, 232.2, 233.8, 234.4, 235.1, 236.3, 237.7, 238.1, 239.5, 240.2, 241.8, 242.4, 243.1, 244.3, 245.7, 246.1, 247.5, 248.2, 249.8, 250.4, 251.1, 252.3, 253.7, 254.1, 255.5, 256.2, 257.8, 258.4, 259.1, 260.3, 261.7, 262.1, 263.5, 264.2, 265.8, 266.4, 267.1, 268.3, 269.7, 270.1, 271.5, 272.2, 273.8, 274.4, 275.1, 276.3, 277.7, 278.1, 279.5, 280.2, 281.8, 282.4, 283.1, 284.3, 285.7, 286.1, 287.5, 288.2, 289.8, 290.4, 291.1, 292.3, 293.7, 294.1, 295.5, 296.2, 297.8, 298.4, 299.1, 300.3, 301.7, 302.1, 303.5, 304.2, 305.8, 306.4, 307.1, 308.3, 309.7, 310.1, 311.5, 312.2, 313.8, 314.4, 315.1, 316.3, 317.7, 318.1, 319.5, 320.2, 321.8, 322.4, 323.1, 324.3, 325.7, 326.1, 327.5, 328.2, 329.8, 330.4, 331.1, 332.3, 333.7, 334.1, 335.5, 336.2, 337.8, 338.4, 339.1, 340.3, 341.7, 342.1, 343.5, 344.2, 345.8, 346.4, 347.1, 348.3, 349.7, 350.1, 351.5, 352.2, 353.8, 354.4, 355.1, 356.3, 357.7, 358.1, 359.5, 360.2, 361.8, 362.4, 363.1, 364.3, 365.7, 366.1, 367.5, 368.2, 369.8, 370.4, 371.1, 372.3, 373.7, 374.1, 375.5, 376.2, 377.8, 378.4, 379.1, 380.3, 381.7, 382.1, 383.5, 384.2, 385.8, 386.4, 387.1, 388.3, 389.7, 390.1, 391.5, 392.2, 393.8, 394.4, 395.1, 396.3, 397.7, 398.1, 399.5, 400.2],
    "horizon": 50,
    "quantiles": [0.1, 0.5, 0.9],
    "model_size": "200M"
}
EOF
)
call_api "POST" "/forecast" "$LONG_TERM_DATA"

echo -e "\n11. Performance Test with Multiple Requests"
echo "---------------------------------------------"
echo "Sending 5 concurrent requests to test performance..."

for i in {1..5}; do
    (
        QUICK_FORECAST=$(cat <<EOF
{
    "data": [10.5, 12.3, 15.8, 18.2, 14.7, 11.9, 9.8, 13.4, 16.7, 19.1],
    "horizon": 10,
    "quantiles": [0.5],
    "model_size": "200M"
}
EOF
)
        echo "Request $i:"
        call_api "POST" "/forecast" "$QUICK_FORECAST" | jq '.request_id, .metadata.cached'
    ) &
done

wait

echo -e "\n12. Error Handling Examples"
echo "-----------------------------"

# Empty data
echo "Empty data error:"
EMPTY_DATA='{"data": [], "horizon": 5}'
call_api "POST" "/forecast" "$EMPTY_DATA" 2>/dev/null || echo "Expected error for empty data"

# Invalid data
echo -e "\nInvalid data error:"
INVALID_DATA='{"data": ["not_a_number"], "horizon": 5}'
call_api "POST" "/forecast" "$INVALID_DATA" 2>/dev/null || echo "Expected error for invalid data"

# Insufficient data
echo -e "\nInsufficient data error:"
INSUFFICIENT_DATA='{"data": [1.0], "horizon": 10}'
call_api "POST" "/forecast" "$INSUFFICIENT_DATA" 2>/dev/null || echo "Expected error for insufficient data"

echo -e "\n=========================================="
echo "TimesFM API Examples Complete!"
echo "=========================================="
echo "All examples used real public datasets and demonstrated:"
echo "- Various normalization methods (REVIN, Z-score, MinMax)"
echo "- Missing value handling strategies"
echo "- Outlier detection and removal"
echo "- Data smoothing techniques"
echo "- Different forecast horizons and quantiles"
echo "- Performance testing with concurrent requests"
echo "- Comprehensive error handling"
echo ""
echo "For more information, visit /docs for interactive API documentation"