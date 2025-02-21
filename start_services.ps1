# Check for existing instances
$comfyUIInstance = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*main.py*" }
$midasInstance = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*app.py*" }

if ($comfyUIInstance -or $midasInstance) {
    Write-Host "Existing instances found. Terminating..."
    $comfyUIInstance | Stop-Process -Force
    $midasInstance | Stop-Process -Force
    Start-Sleep -Seconds 2
}

# Check if Ollama is running
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "Ollama is not running. Starting Ollama..."
    Start-Process -FilePath "ollama" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# Define paths for virtual environments and project directories
$comfyui_venv = "C:\AIapps\ComfyUI_venv\Scripts\Activate.ps1"
$midas_venv = "C:\AIapps\MIDAS_venv\Scripts\Activate.ps1"
$comfyui_path = Join-Path $PSScriptRoot "ComfyUI"

# Start ComfyUI server in its virtual environment
Start-Process powershell -ArgumentList '-NoExit', '-Command', "& '$comfyui_venv'; cd '$comfyui_path'; python main.py" 

# Wait for ComfyUI to initialize
Start-Sleep -Seconds 10

# Start MIDAS application in its virtual environment
Start-Process powershell -ArgumentList '-NoExit', '-Command', "& '$midas_venv'; cd '$PSScriptRoot'; python app.py"
