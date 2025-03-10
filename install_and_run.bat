@echo off
REM Ensure running as administrator
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
"%temp%\getadmin.vbs"
exit /B

:gotAdmin
REM Remove temporary file
if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )

REM Change to script directory
cd /d "%~dp0"

echo [%date% %time%] Starting installation and setup process...

REM Check and install Git, Pip, and Python 3.11 if not already installed
where git >nul 2>&1 || (
    echo [%date% %time%] Installing Git...
    winget install --id Git.Git -e
    echo [%date% %time%] Git installation complete.
)
where python >nul 2>&1 || (
    echo [%date% %time%] Installing Python 3.11...
    winget install --id Python.Python.3.11 -e
    echo [%date% %time%] Python 3.11 installation complete.
)

REM Check and install Ollama if not already installed
where ollama >nul 2>&1 || (
    echo [%date% %time%] Installing Ollama...
    curl -L https://ollama.ai/download/OllamaSetup.exe -o OllamaSetup.exe
    OllamaSetup.exe /S
    echo [%date% %time%] Ollama installation complete.
)

REM Pull basic models if not already present
echo [%date% %time%] Checking and pulling Ollama models...
ollama list | findstr "llama3.1" >nul 2>&1 || (
    echo [%date% %time%] Pulling llama3.1 model...
    ollama pull llama3.1
)
ollama list | findstr "phi3.5" >nul 2>&1 || (
    echo [%date% %time%] Pulling phi3.5 model...
    ollama pull phi3.5
)
ollama list | findstr "mistral" >nul 2>&1 || (
    echo [%date% %time%] Pulling mistral model...
    ollama pull mistral
)

REM Download ComfyUI if not already present
if not exist ComfyUI (
    echo [%date% %time%] Downloading ComfyUI...
    git clone https://github.com/comfyanonymous/ComfyUI.git
    echo [%date% %time%] ComfyUI download complete.
    
    echo [%date% %time%] Installing ComfyUI Manager...
    if not exist ComfyUI\custom_nodes mkdir ComfyUI\custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI\custom_nodes\ComfyUI-Manager
    echo [%date% %time%] ComfyUI Manager installation complete.
)

REM Set up MIDAS Python 3.11 virtual environment if not already set up
if not exist venvs\midas_venv (
    echo [%date% %time%] Setting up MIDAS virtual environment...
    for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i
    "%PYTHON_PATH%" -m venv venvs\midas_venv
    call venvs\midas_venv\Scripts\activate.bat
    pip install -r requirements.txt
    call venvs\midas_venv\Scripts\deactivate.bat
    echo [%date% %time%] MIDAS virtual environment setup complete.
)

REM Set up virtual environments directory
if not exist venvs mkdir venvs

REM Set up ComfyUI Python virtual environment if not already set up
if not exist venvs\comfyui_venv (
    echo [%date% %time%] Setting up ComfyUI virtual environment...
    cd ComfyUI
    for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i
    "%PYTHON_PATH%" -m venv ..\venvs\comfyui_venv
    call ..\venvs\comfyui_venv\Scripts\activate.bat
    pip install -r requirements.txt
    call ..\venvs\comfyui_venv\Scripts\deactivate.bat
    cd ..
    echo [%date% %time%] ComfyUI virtual environment setup complete.
)

REM Download SDXL model if not already present
if not exist ComfyUI\models\checkpoints\sdXL_v10.safetensors (
    echo [%date% %time%] Downloading SDXL model...
    curl -L https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -o ComfyUI\models\checkpoints\sdXL_v10.safetensors
    echo [%date% %time%] SDXL model download complete.
)

echo [%date% %time%] Setup complete.

REM Run all services and open MIDAS in browser
echo [%date% %time%] Starting MIDAS services...
cd %~dp0
start powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0start_services.ps1'"

REM Wait for services to start
echo [%date% %time%] Waiting for services to start...
timeout /t 20 /nobreak

REM Start custom browser
echo [%date% %time%] Starting MIDAS browser...
start "" "%~dp0midas_browser.exe"

REM Wait for browser to close
:wait_for_browser
tasklist | find "midas_browser.exe" > nul
if %errorlevel% equ 0 (
    timeout /t 5 /nobreak > nul
    goto wait_for_browser
) else (
    echo [%date% %time%] Browser closed. Stopping services...
    taskkill /F /IM python.exe /FI "WINDOWTITLE eq *app.py*"
    taskkill /F /IM python.exe /FI "WINDOWTITLE eq *main.py*"
    echo [%date% %time%] All services stopped.
)

echo [%date% %time%] Script execution complete.
