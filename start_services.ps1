# Check if Ollama is running
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "Ollama is not running. Starting Ollama..."
    Start-Process -FilePath "ollama" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# Start ComfyUI server
$comfyui_path = Join-Path $PSScriptRoot "ComfyUI"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "python main.py" -WorkingDirectory $comfyui_path

# Wait for ComfyUI to initialize
Start-Sleep -Seconds 10

# Start MIDAS application
Start-Process powershell -ArgumentList '-NoExit', '-Command', "python app.py" -WorkingDirectory $PSScriptRoot
