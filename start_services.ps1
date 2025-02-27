# Ensure we're in the correct directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Ensure script stops on first error
$ErrorActionPreference = 'Stop'

# Logging function
function Write-ColorLog {
    param([string]$Message, [string]$Color = 'Green')
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Cleanup {
    # Get all python processes
    $pythonProcesses = Get-Process | Where-Object { $_.ProcessName -match 'python(?:w)?' }

    foreach ($proc in $pythonProcesses) {
        try {
            # Check if process is running MIDAS or ComfyUI
            if ($proc.CommandLine -match 'app.py|main.py') {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
        } catch {
            # Silently continue on errors
            continue
        }
    }
}

# Register cleanup function to run on script exit
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Cleanup }

try {
    # Check for existing instances
    Write-ColorLog "Checking for existing instances..." "Yellow"
    $comfyUIInstance = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*main.py*" }
    $midasInstance = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*app.py*" }

    if ($comfyUIInstance -or $midasInstance) {
        Write-ColorLog "Existing instances found. Terminating..." "Yellow"
        $comfyUIInstance | Stop-Process -Force
        $midasInstance | Stop-Process -Force
        Start-Sleep -Seconds 2
    }

    # Check if Ollama is running
    Write-ColorLog "Checking Ollama service..." "Yellow"
    $ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaProcess) {
        Write-ColorLog "Ollama is not running. Starting Ollama..." "Yellow"
        Start-Process -FilePath "ollama" -WindowStyle Hidden
        Start-Sleep -Seconds 5
    }

    # Define paths for virtual environments and project directories
    $comfyui_venv = Join-Path $scriptPath "venvs\comfyui_venv\Scripts\Activate.ps1"
    $midas_venv = Join-Path $scriptPath "venvs\midas_venv\Scripts\Activate.ps1"
    $comfyui_path = Join-Path $scriptPath "ComfyUI"

    # Verify paths exist
    if (-not (Test-Path $comfyui_venv)) { throw "ComfyUI venv not found at: $comfyui_venv" }
    if (-not (Test-Path $midas_venv)) { throw "MIDAS venv not found at: $midas_venv" }
    if (-not (Test-Path $comfyui_path)) { throw "ComfyUI directory not found at: $comfyui_path" }

    # Start ComfyUI server in its virtual environment
    Write-ColorLog "Starting ComfyUI server..." "Green"
    $comfyUIWindow = Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$scriptPath'; & '$comfyui_venv'; Set-Location '$comfyui_path'; python main.py; Read-Host 'Press Enter to close'"
    ) -PassThru -WindowStyle Normal

    # Wait for ComfyUI to initialize
    Write-ColorLog "Waiting for ComfyUI to initialize..." "Green"
    Start-Sleep -Seconds 20

    # Start MIDAS application in its virtual environment
    Write-ColorLog "Starting MIDAS application..." "Green"
    $midasProcess = Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$scriptPath'; & '$midas_venv'; python app.py; Read-Host 'Press Enter to close'"
    ) -PassThru -WindowStyle Normal

    Write-ColorLog "Waiting for MIDAS application to initialize..." "Green"
    Start-Sleep -Seconds 10

    Write-ColorLog "All services started successfully." "Green"
    Write-ColorLog "Press Ctrl+C to terminate all services." "Yellow"
    
    # Keep the script running and monitor child processes
    while ($true) {
        Start-Sleep -Seconds 1
        if (-not (Get-Process -Id $comfyUIWindow.Id -ErrorAction SilentlyContinue) -or
            -not (Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*app.py*" })) {
            Write-ColorLog "One or more services have stopped. Terminating all services..." "Red"
            break
        }
    }
} catch {
    Write-ColorLog "Error: $_" "Red"
    Write-ColorLog "Stack Trace: $($_.ScriptStackTrace)" "Red"
    Write-ColorLog "Press any key to exit..." "Red"
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
} finally {
    # Cleanup
    Get-Process -Name "python" | Where-Object { $_.CommandLine -like "*main.py*" -or $_.CommandLine -like "*app.py*" } | Stop-Process -Force
    Write-ColorLog "Services stopped." "Yellow"
    Start-Sleep -Seconds 2
}
