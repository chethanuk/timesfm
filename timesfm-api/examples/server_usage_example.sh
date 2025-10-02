#!/bin/bash

# TimesFM API Server Usage Examples
# This script demonstrates various ways to use the startup script

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}TimesFM API Server Usage Examples${NC}"
echo "=================================="
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
STARTUP_SCRIPT="${SCRIPT_DIR}/scripts/start_server.sh"

echo -e "${YELLOW}1. Server Management Commands${NC}"
echo "-------------------------------"

echo "Start server:"
echo "  ${STARTUP_SCRIPT} start"
echo

echo "Start server with custom port:"
echo "  ${STARTUP_SCRIPT} start PORT=8080"
echo

echo "Start server with debug logging:"
echo "  ${STARTUP_SCRIPT} start LOG_LEVEL=debug"
echo

echo "Start server with multiple workers:"
echo "  ${STARTUP_SCRIPT} start WORKERS=4"
echo

echo "Check server status:"
echo "  ${STARTUP_SCRIPT} status"
echo

echo "Stop server:"
echo "  ${STARTUP_SCRIPT} stop"
echo

echo "Restart server:"
echo "  ${STARTUP_SCRIPT} restart"
echo

echo "Show logs:"
echo "  ${STARTUP_SCRIPT} logs"
echo "  ${STARTUP_SCRIPT} logs 100"
echo "  ${STARTUP_SCRIPT} logs -f"
echo

echo -e "${YELLOW}2. Environment Variables${NC}"
echo "----------------------------"

cat << 'EOF'
HOST            Server host (default: 0.0.0.0)
PORT            Server port (default: 8000)
WORKERS         Number of worker processes (default: 1)
LOG_LEVEL       Log level (debug, info, warning, error)
RELOAD          Enable auto-reload for development (true/false)
TIMEOUT         Request timeout in seconds (default: 30)
MAX_REQUESTS    Max requests per worker before restart
MAX_REQUESTS_JITTER Jitter for max requests

EOF

echo -e "${YELLOW}3. Production Deployment Examples${NC}"
echo "------------------------------------"

echo "Production start with multiple workers:"
echo "  ${STARTUP_SCRIPT} start WORKERS=4 LOG_LEVEL=info TIMEOUT=60"
echo

echo "Development start with auto-reload:"
echo "  ${STARTUP_SCRIPT} start RELOAD=true LOG_LEVEL=debug"
echo

echo "Custom port and configuration:"
echo "  ${STARTUP_SCRIPT} start PORT=9000 WORKERS=2 MAX_REQUESTS=5000"
echo

echo -e "${YELLOW}4. Log Management${NC}"
echo "-------------------"

echo "View real-time logs:"
echo "  ${STARTUP_SCRIPT} logs -f"
echo

echo "View last 100 lines:"
echo "  ${STARTUP_SCRIPT} logs 100"
echo

echo "Check server health:"
echo "  ${STARTUP_SCRIPT} health"
echo

echo -e "${YELLOW}5. API Endpoints (when server is running)${NC}"
echo "------------------------------------------"

echo "Health check:"
echo "  curl http://localhost:8000/health"
echo

echo "API Documentation:"
echo "  Swagger UI: http://localhost:8000/docs"
echo "  ReDoc: http://localhost:8000/redoc"
echo "  OpenAPI Spec: http://localhost:8000/openapi.json"
echo

echo "Preprocessing methods:"
echo "  curl http://localhost:8000/preprocess/methods"
echo

echo "Sample forecast:"
echo "  curl -X POST http://localhost:8000/forecast \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"data\": [1,2,3,4,5], \"horizon\": 3}'"
echo

echo -e "${YELLOW}6. Monitoring and Troubleshooting${NC}"
echo "-----------------------------------"

echo "Check if server is running:"
echo "  ${STARTUP_SCRIPT} status"
echo

echo "Follow logs for debugging:"
echo "  ${STARTUP_SCRIPT} logs -f"
echo

echo "Test server health:"
echo "  ${STARTUP_SCRIPT} health"
echo "  curl -f http://localhost:8000/health || echo 'Server not responding'"
echo

echo "Check process:"
echo "  ps aux | grep uvicorn"
echo "  lsof -i :8000"
echo

echo -e "${GREEN}All set! Use these commands to manage your TimesFM API server.${NC}"
echo
echo "For complete documentation, visit: http://localhost:8000/docs"