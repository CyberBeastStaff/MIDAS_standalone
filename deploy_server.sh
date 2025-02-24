#!/bin/bash

# Update and install necessary packages
sudo yum update -y
sudo yum install -y git python3 python3-pip curl wget

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models from Ollama
echo "Pulling required models from Ollama..."
for model in llama3.1 codellama phi3.5; do
    if ! ollama pull $model; then
        echo "Failed to pull model: $model. Please check your internet connection and try again."
        exit 1
    fi
done

# Set up virtual environment for MIDAS
echo "Setting up virtual environment for MIDAS..."
cd MIDAS_standaloneapp
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download SDXL model for ComfyUI
echo "Downloading SDXL model for ComfyUI..."
mkdir -p comfyui/models/checkpoints
if ! wget -O comfyui/models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors; then
    echo "Failed to download SDXL model. Please check your internet connection and try again."
    exit 1
fi
mv comfyui/models/checkpoints/sd_xl_base_1.0.safetensors comfyui/models/checkpoints/sdXL_v10.safetensors

# Set up virtual environment for ComfyUI
echo "Setting up virtual environment for ComfyUI..."
cd comfyui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the servers
echo "Starting Ollama server..."
ollama serve &

echo "Starting MIDAS server..."
cd ../MIDAS_standaloneapp
python app.py &

echo "Starting ComfyUI server..."
cd ../comfyui
python app.py &

echo "All servers started. Deployment complete!"