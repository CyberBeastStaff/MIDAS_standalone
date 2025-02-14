#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a process is running
check_process() {
    pgrep -f "$1" > /dev/null
    return $?
}

# Function to log messages
log() {
    echo -e "${2:-$GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to start a service
start_service() {
    local service_name=$1
    local command=$2
    local check_pattern=$3

    log "Starting $service_name..." "$YELLOW"
    eval "$command" &
    sleep 5

    if check_process "$check_pattern"; then
        log "$service_name started successfully"
    else
        log "Failed to start $service_name" "$RED"
        exit 1
    fi
}

# Create necessary directories
mkdir -p logs

# Kill existing processes
log "Checking for existing processes..." "$YELLOW"
pkill -f "ollama serve" > /dev/null 2>&1
pkill -f "python.*main.py" > /dev/null 2>&1
pkill -f "python.*app.py" > /dev/null 2>&1
sleep 2

# Start Ollama
start_service "Ollama" "ollama serve > logs/ollama.log 2>&1" "ollama serve"

# Start ComfyUI
if [ -d "ComfyUI" ]; then
    cd ComfyUI
    if [ -d "venv" ]; then
        source venv/bin/activate
        start_service "ComfyUI" "python main.py > ../logs/comfyui.log 2>&1" "python.*main.py"
        deactivate
    else
        log "ComfyUI virtual environment not found" "$RED"
        exit 1
    fi
    cd ..
else
    log "ComfyUI directory not found" "$RED"
    exit 1
fi

# Start MIDAS
if [ -d "venv" ]; then
    source venv/bin/activate
    start_service "MIDAS" "python app.py > logs/midas.log 2>&1" "python.*app.py"
else
    log "MIDAS virtual environment not found" "$RED"
    exit 1
fi

# Monitor services
log "All services started. Monitoring logs..." "$YELLOW"
tail -f logs/*.log

# Cleanup function
cleanup() {
    log "Shutting down services..." "$YELLOW"
    pkill -f "ollama serve"
    pkill -f "python.*main.py"
    pkill -f "python.*app.py"
    log "All services stopped"
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

# Keep script running
while true; do
    if ! check_process "ollama serve" || ! check_process "python.*main.py" || ! check_process "python.*app.py"; then
        log "One or more services have stopped unexpectedly" "$RED"
        cleanup
    fi
    sleep 10
done
