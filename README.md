# MIDAS 2.0: Machine Intelligence Deployment and Automation System

## Overview
MIDAS is an advanced AI-powered chatbot and image generation application that provides a seamless, interactive experience with multiple AI models and capabilities. It integrates local language models via Ollama with powerful image generation capabilities through ComfyUI.

## Features
- **Multi-model Chat Interface**: Chat with various AI models including Phi3.5, Mistral, Llama3.1, and custom models
- **Thinking Process Visibility**: See the AI's reasoning process before getting the final answer
- **SDXL Image Generation**: Create high-quality images using text prompts with customizable parameters
- **Conversation History**: Save, search, and continue previous conversations
- **Bot Configuration Management**: Create and customize AI assistants with specific personalities and capabilities
- **Workflow Support**: Use predefined ComfyUI workflows for specialized image generation
- **Dynamic Model Loading**: Automatically detects and loads available models from Ollama
- **Markdown Support**: Rich text formatting in chat responses

## Components

MIDAS consists of three main components:
1. **MIDAS Interface**: Main application interface (included in this repository)
2. **Ollama**: Local LLM server (installed separately)
3. **ComfyUI**: Image generation backend (installed separately, with ComfyUI Manager)

## Prerequisites
- Python 3.10 or 3.11 (3.12 not yet fully supported)
- Ollama
- ComfyUI (Required for image generation)
- Git

## Installation

### Automated Installation (Windows)

The easiest way to install MIDAS on Windows is using the provided installation script:

1. Run `install_and_run.bat` as administrator
2. The script will:
   - Check for and install Python if needed
   - Install Git if needed
   - Download and install Ollama
   - Clone ComfyUI repository and install ComfyUI Manager
   - Set up Python virtual environment
   - Install all required dependencies
   - Pull necessary Ollama models

### Manual Installation

#### Required Dependencies

Before installing MIDAS manually, ensure you have the following components installed:

1. **Ollama**
   - Windows/Mac: Download from [Ollama's website](https://ollama.ai)
   - Required models: phi3.5, mistral, llama3.1

2. **ComfyUI** (Not included in repository)
   - Clone separately: `git clone https://github.com/comfyanonymous/ComfyUI.git`
   - Install ComfyUI Manager: `git clone https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI\custom_nodes\ComfyUI-Manager`
   - Install in the same parent directory as MIDAS
   - Follow ComfyUI's installation guide for model setup

#### Windows Manual Installation
##### 1. Clone the Repository
```bash
git clone https://github.com/CyberBeastStaff/MIDAS_standalone.git
cd MIDAS_standalone
```

##### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

##### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

##### 4. Configure Ollama
Ensure Ollama is installed and models are pulled:
```bash
ollama pull phi3.5
ollama pull mistral
ollama pull llama3.1
```

##### 5. ComfyUI Setup
```bash
# Navigate to parent directory
cd ..

# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git

# Install ComfyUI Manager
mkdir ComfyUI\custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI\custom_nodes\ComfyUI-Manager

# Setup ComfyUI
cd ComfyUI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Download required models through ComfyUI Manager interface
# after starting ComfyUI
```

#### macOS Installation

##### Prerequisites
1. Install Homebrew (Package Manager)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install Python and Git
```bash
brew install python@3.10
brew install git
```

##### Installing Ollama
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

##### Installing MIDAS
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

##### Installing ComfyUI
```bash
# Navigate to parent directory
cd ..

# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git

# Install ComfyUI Manager
mkdir -p ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI/custom_nodes/ComfyUI-Manager

# Setup ComfyUI
cd ComfyUI
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For M1/M2/M3 Macs (Apple Silicon)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Download required models through ComfyUI Manager interface
# after starting ComfyUI
```

##### Starting Services on Mac
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

## Using MIDAS

### Starting the Application

#### Windows
1. Run `start_services.ps1` PowerShell script:
   - Right-click and select "Run with PowerShell" or
   - Open PowerShell and run `.\start_services.ps1`

#### Mac
1. Run `./start_services.sh` in Terminal

### Chat Interface

1. Open your browser and navigate to `http://localhost:7860`
2. Click "New Chat" to start a conversation
3. Select a model from the dropdown menu
4. Type your message and press Enter or click Send

### Image Generation Commands

MIDAS supports SDXL image generation with various customization options using command flags:

#### Basic Usage
```
Generate an image of a mountain landscape
```

#### Command Flags
Add these flags after your prompt to customize the image:

- `--ar [preset]`: Aspect ratio preset
  - `square`: 1024×1024 (default)
  - `portrait`: 832×1216
  - `landscape`: 1216×832
  - `wide`: 1344×768
  - `tall`: 768×1344

- `--style [preset]`: Style preset
  - `photo`: Detailed photograph
  - `anime`: Anime style
  - `painting`: Digital painting
  - `sketch`: Pencil sketch
  - `3d`: 3D render
  - `cinematic`: Cinematic shot

- `--quality [value]`: Quality level (0.1-2.0, default: 1.0)
- `--steps [value]`: Number of generation steps (1-100, default: 30)
- `--cfg [value]`: CFG scale (1.0-20.0, default: 7.0)
- `--seed [value]`: Specific seed for reproducible results
- `--neg [text]`: Negative prompt to specify what to avoid

#### Examples

```
A beautiful sunset over mountains --ar landscape --style photo --quality 1.5

A cute anime character with blue hair --ar portrait --style anime --steps 40

A futuristic city with flying cars --style 3d --neg "blurry, low quality, distorted"
```

### Using Workflows

MIDAS supports custom ComfyUI workflows for specialized image generation:

1. Select "Workflow: Flux 1" or other available workflows from the model dropdown
2. Enter your prompt with any command flags
3. The system will use the specified workflow for image generation

## Troubleshooting

### Common Issues

1. **Ollama Not Running**
   - Error: "Failed to connect to Ollama"
   - Solution: Ensure Ollama is running with `ollama serve`

2. **ComfyUI Not Running**
   - Error: "Failed to connect to ComfyUI"
   - Solution: Check ComfyUI logs and ensure it's running on port 8188

3. **Missing Models**
   - Error: "Model not found"
   - Solution: Pull the required model with `ollama pull [model_name]`

4. **CUDA/GPU Issues**
   - Solution: Update GPU drivers and ensure CUDA is properly installed

### Logs
Check the logs in the `logs` directory for detailed error information:
- `ollama.log`: Ollama server logs
- `comfyui.log`: ComfyUI server logs
- `midas.log`: MIDAS application logs

## Directory Structure
```
Parent Directory
├── MIDAS_standalone/    # This repository
│   ├── app.py           # Main application
│   ├── requirements.txt # Python dependencies
│   ├── config.ini       # Configuration file
│   ├── start_services.ps1 # Windows startup script
│   ├── start_services.sh  # Mac startup script
│   ├── install_and_run.bat # Windows installation script
│   ├── conversations.db # Conversation database
│   ├── bots.db          # Bot configurations
│   └── workflows/       # ComfyUI workflow definitions
├── ComfyUI/             # ComfyUI repository (installed separately)
│   ├── ...
│   └── custom_nodes/    # ComfyUI extensions
│       └── ComfyUI-Manager/ # ComfyUI Manager
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
