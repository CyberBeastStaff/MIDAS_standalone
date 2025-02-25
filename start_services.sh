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
#start_service "Ollama" "ollama serve > logs/ollama.log 2>&1" "ollama serve"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Start ComfyUI server
cd "$SCRIPT_DIR/ComfyUI"
if [ -d "venv" ]; then
    source venv/bin/activate
    start_service "ComfyUI" "python main.py > ../logs/comfyui.log 2>&1" "python.*main.py"
    deactivate
else
    log "ComfyUI virtual environment not found" "$RED"
    exit 1
fi
# Download SDXL model if it doesn't exist
SDXL_MODEL_PATH="$SCRIPT_DIR/ComfyUI/models/checkpoints/sdXL_v10.safetensors"
if [ ! -f "$SDXL_MODEL_PATH" ]; then
    log "Downloading SDXL model..." "$YELLOW"
    mkdir -p "$(dirname "$SDXL_MODEL_PATH")"
    TEMP_FILE="$(mktemp)"
    wget -O "$TEMP_FILE" "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    if [ $? -eq 0 ]; then
        mv "$TEMP_FILE" "$SDXL_MODEL_PATH"
        log "SDXL model downloaded and renamed successfully"
    else
        log "Failed to download SDXL model" "$RED"
        rm -f "$TEMP_FILE"
        exit 1
    fi
else
    log "SDXL model already exists"
fi

# Start MIDAS
cd "$SCRIPT_DIR"
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
