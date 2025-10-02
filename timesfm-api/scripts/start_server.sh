#!/bin/bash

# TimesFM API Production Startup Script
# This script starts the TimesFM API server in background with proper logging

set -euo pipefail

# Configuration
API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${API_DIR}/logs"
PID_FILE="${API_DIR}/timesfm-api.pid"
LOG_FILE="${API_DIR}/logs/timesfm-api.log"
ACCESS_LOG="${API_DIR}/logs/access.log"
ERROR_LOG="${API_DIR}/logs/error.log"

# Default settings
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}
RELOAD=${RELOAD:-false}
MAX_REQUESTS=${MAX_REQUESTS:-1000}
MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}
TIMEOUT=${TIMEOUT:-30}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${ERROR_LOG}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

# Function to check if server is already running
is_server_running() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid=$(cat "${PID_FILE}")
        if ps -p "${pid}" > /dev/null 2>&1; then
            return 0
        else
            rm -f "${PID_FILE}"
            return 1
        fi
    fi
    return 1
}

# Function to stop the server
stop_server() {
    if is_server_running; then
        local pid=$(cat "${PID_FILE}")
        log_info "Stopping TimesFM API server (PID: ${pid})..."

        # Send SIGTERM for graceful shutdown
        kill -TERM "${pid}" 2>/dev/null || true

        # Wait for graceful shutdown
        local count=0
        while ps -p "${pid}" > /dev/null 2>&1 && [[ ${count} -lt 30 ]]; do
            sleep 1
            ((count++))
        done

        # Force kill if still running
        if ps -p "${pid}" > /dev/null 2>&1; then
            log_warn "Server did not stop gracefully, forcing shutdown..."
            kill -KILL "${pid}" 2>/dev/null || true
        fi

        rm -f "${PID_FILE}"
        log_success "Server stopped successfully"
    else
        log_warn "Server is not running"
    fi
}

# Function to check server health
check_server_health() {
    local max_attempts=30
    local attempt=1

    log_info "Checking server health..."

    while [[ ${attempt} -le ${max_attempts} ]]; do
        if curl -s -f "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
            log_success "Server is healthy and responding"
            return 0
        fi

        log_info "Health check attempt ${attempt}/${max_attempts} - waiting..."
        sleep 2
        ((attempt++))
    done

    log_error "Server health check failed after ${max_attempts} attempts"
    return 1
}

# Function to show server status
show_status() {
    echo "=== TimesFM API Server Status ==="
    echo

    if is_server_running; then
        local pid=$(cat "${PID_FILE}")
        echo "Status: ${GREEN}Running${NC}"
        echo "PID: ${pid}"
        echo "URL: http://${HOST}:${PORT}"
        echo "Logs: ${LOG_FILE}"

        # Show process info
        echo
        echo "Process Information:"
        ps -p "${pid}" -o pid,ppid,cmd,etime,pcpu,pmem --no-headers || echo "Process info not available"

        # Show recent logs
        echo
        echo "Recent Logs (last 10 lines):"
        tail -n 10 "${LOG_FILE}" 2>/dev/null || echo "No logs available"

    else
        echo "Status: ${RED}Stopped${NC}"
        echo "URL: http://${HOST}:${PORT}"
        echo "Logs: ${LOG_FILE}"
    fi
}

# Function to show logs
show_logs() {
    local lines=${1:-50}
    local follow=${2:-false}

    if [[ ! -f "${LOG_FILE}" ]]; then
        echo "Log file not found: ${LOG_FILE}"
        return 1
    fi

    echo "=== TimesFM API Server Logs ==="
    echo "Log file: ${LOG_FILE}"
    echo "Showing last ${lines} lines"
    echo

    if [[ "${follow}" == "true" ]]; then
        tail -f -n "${lines}" "${LOG_FILE}"
    else
        tail -n "${lines}" "${LOG_FILE}"
    fi
}

# Function to setup log directory
setup_logs() {
    mkdir -p "${LOG_DIR}"

    # Create log files if they don't exist
    touch "${LOG_FILE}" "${ACCESS_LOG}" "${ERROR_LOG}"

    # Set up log rotation
    if command -v logrotate >/dev/null 2>&1; then
        cat > "${API_DIR}/logrotate.conf" << EOF
${LOG_FILE} ${ACCESS_LOG} ${ERROR_LOG} {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
    postrotate
        # Send USR1 to server to reopen logs if running
        if [[ -f "${PID_FILE}" ]]; then
            pid=\$(cat "${PID_FILE}")
            if ps -p "\$pid" > /dev/null 2>&1; then
                kill -USR1 "\$pid" 2>/dev/null || true
            fi
        fi
    endscript
}
EOF
    fi
}

# Function to start the server
start_server() {
    if is_server_running; then
        log_error "Server is already running (PID: $(cat "${PID_FILE}"))"
        echo "Use '$0 status' to check the server status"
        echo "Use '$0 stop' to stop the server first"
        return 1
    fi

    log_info "Starting TimesFM API server..."
    log_info "Configuration:"
    log_info "  Host: ${HOST}"
    log_info "  Port: ${PORT}"
    log_info "  Workers: ${WORKERS}"
    log_info "  Log Level: ${LOG_LEVEL}"
    log_info "  Reload: ${RELOAD}"
    log_info "  Timeout: ${TIMEOUT}s"
    log_info "  Max Requests: ${MAX_REQUESTS}"

    # Setup logging
    setup_logs

    # Change to API directory
    cd "${API_DIR}"

    # Prepare uvicorn command
    local uvicorn_cmd=(
        uvicorn
        app:app
        --host "${HOST}"
        --port "${PORT}"
        --workers "${WORKERS}"
        --log-level "${LOG_LEVEL}"
        --timeout-keep-alive "${TIMEOUT}"
        --access-log
    )

    # Add development options if reload is enabled
    if [[ "${RELOAD}" == "true" ]]; then
        uvicorn_cmd+=(--reload)
    fi

    # Export environment variables
    export PYTHONPATH="${API_DIR}:${PYTHONPATH:-}"
    export TIMESFM_LOG_LEVEL="${LOG_LEVEL}"
    export TIMESFM_LOG_FILE="${LOG_FILE}"

    # Start the server
    log_info "Executing: ${uvicorn_cmd[*]}"
    "${uvicorn_cmd[@]}" 1>> "${LOG_FILE}" 2>&1 &

    # Capture PID
    local pid=$!
    echo "${pid}" > "${PID_FILE}"

    log_info "Server started with PID: ${pid}"
    log_info "Waiting for server to be ready..."

    # Wait for server to be ready
    if check_server_health; then
        log_success "TimesFM API server is ready!"
        echo
        echo "🚀 Server Information:"
        echo "  URL: http://${HOST}:${PORT}"
        echo "  PID: ${pid}"
        echo "  Logs: ${LOG_FILE}"
        echo "  PID File: ${PID_FILE}"
        echo
        echo "📚 Documentation:"
        echo "  Swagger UI: http://${HOST}:${PORT}/docs"
        echo "  ReDoc: http://${HOST}:${PORT}/redoc"
        echo "  OpenAPI Spec: http://${HOST}:${PORT}/openapi.json"
        echo
        echo "🔧 Management Commands:"
        echo "  Status: $0 status"
        echo "  Stop: $0 stop"
        echo "  Logs: $0 logs"
        echo "  Follow Logs: $0 logs -f"
        echo

        return 0
    else
        log_error "Server failed to start properly"
        stop_server
        return 1
    fi
}

# Function to show help
show_help() {
    cat << EOF
TimesFM API Server Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start           Start the server in background
    stop            Stop the running server
    restart         Restart the server
    status          Show server status and recent logs
    logs [LINES]    Show recent logs (default: 50 lines)
    logs -f         Follow logs in real-time
    health          Check server health
    help            Show this help message

Environment Variables:
    HOST            Server host (default: 0.0.0.0)
    PORT            Server port (default: 8000)
    WORKERS         Number of worker processes (default: 1)
    LOG_LEVEL       Log level (debug, info, warning, error)
    RELOAD          Enable auto-reload for development (true/false)
    TIMEOUT         Request timeout in seconds (default: 30)
    MAX_REQUESTS    Max requests per worker before restart
    MAX_REQUESTS_JITTER Jitter for max requests

Examples:
    $0 start                    # Start with default settings
    $0 start PORT=8080          # Start on port 8080
    $0 start LOG_LEVEL=debug    # Start with debug logging
    $0 restart                  # Restart the server
    $0 status                   # Show server status
    $0 logs 100                 # Show last 100 log lines
    $0 logs -f                  # Follow logs in real-time

EOF
}

# Main script logic
main() {
    case "${1:-help}" in
        start)
            start_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            stop_server
            sleep 2
            start_server
            ;;
        status)
            show_status
            ;;
        logs)
            if [[ "${2:-}" == "-f" ]]; then
                show_logs "${3:-50}" true
            else
                show_logs "${2:-50}" false
            fi
            ;;
        health)
            if check_server_health; then
                echo "✅ Server is healthy"
                exit 0
            else
                echo "❌ Server health check failed"
                exit 1
            fi
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: ${1}"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'stop_server' SIGINT SIGTERM

# Run main function with all arguments
main "$@"