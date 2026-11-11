# ==========================================================
# AI_FaceRecognitor Launcher
# ==========================================================
# Sets isolation env vars, points onnxruntime at the CUDA 12 DLLs
# shipped with local torch, then runs find_faces.py with all args.
#
# Usage examples:
#   .\run_faces.ps1 --help
#   .\run_faces.ps1 --search-dir O:\PinterestDownloads\vip_board `
#       --target-dir O:\PinterestDownloads\vip_board\cailee_spaeny `
#       --samples "a.jpg,b.jpg"
# ==========================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ScriptPath = $PSScriptRoot
Set-Location $ScriptPath

# isolation
$env:UV_CACHE_DIR = Join-Path $ScriptPath ".cache\uv"
$env:HF_HOME = Join-Path $ScriptPath ".cache\huggingface"
$env:TMP = Join-Path $ScriptPath ".cache\tmp"
$env:TEMP = $env:TMP
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONIOENCODING = "utf-8"

$PythonExe = Join-Path $ScriptPath "face_env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Environment not found. Run FaceRecognitor_portable.bat/.sh/.ps1 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

& $PythonExe (Join-Path $ScriptPath "find_faces.py") @args

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
