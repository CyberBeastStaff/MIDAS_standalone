# MIDAS: Machine Intelligence Deployment and Automation System

## Overview
MIDAS is an advanced AI-powered chatbot and image generation application that provides a seamless, interactive experience with multiple AI models and capabilities.

## Features
- Multi-model chat interface
- SDXL Image Generation
- Conversation History
- Bot Configuration Management
- Flexible Prompt Engineering

## Prerequisites
- Python 3.8+
- Ollama
- ComfyUI (Optional for image generation)

## Installation

### Windows Installation
### 1. Clone the Repository
```bash
git clone https://github.com/CyberBeastStaff/MIDAS_standalone.git
cd MIDAS
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Ollama
Ensure Ollama is installed and models are pulled:
```bash
ollama pull phi3.5
ollama pull mistral
ollama pull llama3.1
```

### 5. (Optional) ComfyUI Setup
If using image generation:
1. Clone ComfyUI
2. Install ComfyUI dependencies
3. Download SDXL models

### macOS Installation

#### Prerequisites
1. Install Homebrew (Package Manager)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install Python and Git
```bash
brew install python@3.10
brew install git
```

#### Installing Ollama
```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# Pull required models
ollama pull phi3.5
ollama pull mistral
ollama pull llama3.1
```

#### Installing MIDAS
```bash
# Clone repository
git clone https://github.com/CyberBeastStaff/MIDAS_standalone.git
cd MIDAS_standalone

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Installing ComfyUI
```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For M1/M2/M3 Macs (Apple Silicon)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

#### Starting Services on Mac
1. Make the startup script executable:
```bash
chmod +x start_services.sh
```

2. Start all services:
```bash
./start_services.sh
```

The script will:
- Start Ollama service
- Launch ComfyUI
- Start MIDAS interface
- Create log files in `logs` directory
- Monitor all services

#### Mac-Specific Notes
- **Apple Silicon (M1/M2/M3)**
  - Native GPU acceleration available
  - Better performance with optimized models
  - Use MPS (Metal Performance Shaders) backend

- **Intel Macs**
  - CPU-only operation
  - May experience slower image generation
  - Limited to CPU-based model inference

- **System Requirements**
  - macOS Monterey (12.0) or later
  - Minimum 16GB RAM recommended
  - 20GB+ free storage space

- **Troubleshooting**
  - Check logs in `logs` directory
  - Ensure all virtual environments are activated
  - Verify Ollama service is running
  - Monitor system resources

## Database
The repository includes test databases with sample conversations and bot configurations. These can be used as initial data or reference for your setup.

### Included Databases
- `conversations.db`: Conversation history
- `bots.db`: Bot configurations
- `documents.db`: Stored document references

**Note**: You can replace or modify these databases as needed for your specific use case.

## Running the Application
```bash
python app.py
```

## Image Generation Commands
Use inline commands for advanced image generation:

```
# Basic generation
A beautiful landscape

# Aspect Ratio
A mountain scene --ar wide

# Style Preset
A cyberpunk city --style cinematic

# Advanced Parameters
Futuristic robot --ar wide --style 3d --steps 50 --cfg 8.0
```

### Command Reference
- `--ar`: Aspect Ratio (square, portrait, landscape, wide, tall)
- `--style`: Image Style (photo, anime, painting, sketch, 3d, cinematic)
- `--steps`: Sampling Steps (1-100)
- `--cfg`: Classifier Free Guidance (1.0-20.0)
- `--quality`: Denoising Strength (0.1-2.0)
- `--seed`: Reproducible Generation
- `--neg`: Negative Prompt

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License
PENDING

## Contact
PENDING
