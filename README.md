# MIDAS: Multi-Intelligent Digital Assistant System

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

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MIDAS.git
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
ollama pull llama2
ollama pull mistral
```

### 5. (Optional) ComfyUI Setup
If using image generation:
1. Clone ComfyUI
2. Install ComfyUI dependencies
3. Download SDXL models

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
[Specify your license here]

## Contact
[Your contact information]
