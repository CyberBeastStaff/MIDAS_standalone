@echo off

REM Check and install Git, Pip, and Python 3.11 if not already installed
where git >nul 2>&1 || (
    echo Installing Git...
    winget install --id Git.Git -e
)
where python >nul 2>&1 || (
    echo Installing Python 3.11...
    winget install --id Python.Python.3.11 -e
)

REM Check and install Ollama if not already installed
where ollama >nul 2>&1 || (
    echo Installing Ollama...
    curl -L https://ollama.ai/download/OllamaSetup.exe -o OllamaSetup.exe
    OllamaSetup.exe /S
)

REM Pull basic models if not already present
ollama list | findstr "llama3.1" >nul 2>&1 || ollama pull llama3.1
ollama list | findstr "phi3.5" >nul 2>&1 || ollama pull phi3.5
ollama list | findstr "mistral" >nul 2>&1 || ollama pull mistral

REM Download ComfyUI if not already present
if not exist ComfyUI (
    echo Downloading ComfyUI...
    git clone https://github.com/comfyanonymous/ComfyUI.git
)
cd ComfyUI

REM Set up ComfyUI Python virtual environment if not already set up
if not exist comfy_venv (
    echo Setting up ComfyUI virtual environment...
    python -m venv comfy_venv
    call comfy_venv\Scripts\activate.bat
    pip install -r requirements.txt
    call comfy_venv\Scripts\deactivate.bat
)

REM Download SDXL model if not already present
if not exist models\sdXL_v10.safetensors (
    echo Downloading SDXL model...
    curl -L https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -o models/sdXL_v10.safetensors
)

cd ..

REM Set up MIDAS Python 3.11 virtual environment if not already set up
if not exist midas_venv311 (
    echo Setting up MIDAS virtual environment...
    "C:\Program Files\Python311\python.exe" -m venv midas_venv311
    call midas_venv311\Scripts\activate.bat
    pip install -r requirements.txt
    call midas_venv311\Scripts\deactivate.bat
)

echo Setup complete.

REM Run all services and open MIDAS in browser
echo Starting MIDAS services...
start cmd /k python ..\app.py
start cmd /k python ..\start_services.py

REM Wait for services to start
timeout /t 5 /nobreak

REM Open MIDAS in default browser
start http://localhost:7860

echo Installation and setup complete! MIDAS is now running and opened in your browser.
REM Kill all services when this window closes
:cleanup
taskkill /F /IM python.exe /FI "WINDOWTITLE eq MIDAS*"
taskkill /F /IM ollama.exe
exit /b

REM Set up cleanup to run when the window is closed
if not "%1"=="CLEANUP" (
    start "" cmd /c "%~f0" CLEANUP
    exit /b
)

REM Wait for this window to close
:wait
timeout /t 1 /nobreak >nul
goto wait
