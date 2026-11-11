# ==========================================================
# AI_FaceRecognitor Portable Installer (idempotent)
# ==========================================================
# Modelled on Trellis2_portable.ps1 (by Soror L.'.L.'. / pytraveler)
#
# Fully IDEMPOTENT: safe to run repeatedly. Every stage checks whether its
# result already exists and skips work if so. No prompts on re-runs.
# Fully SELF-CONTAINED: caches, venv and models live inside the project
# folder; ComfyUI is NOT used or touched. Models are downloaded from the
# official insightface v0.7 release via pycurl (vendored wheel, with a
# urllib fallback).
#
# Usage: powershell -ExecutionPolicy Bypass -File .\FaceRecognitor_portable.ps1
# After install use run_faces.ps1
# ==========================================================

# === Path & Init ===
$ScriptPath = $PSScriptRoot
if (-not $ScriptPath) { $ScriptPath = "." }
Set-Location $ScriptPath
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ==========================================
# === PORTABILITY ISOLATION BLOCK ===
# ==========================================
$CacheDir = Join-Path $ScriptPath ".cache"
$UvCacheDir = Join-Path $CacheDir "uv"
$TmpDir = Join-Path $CacheDir "tmp"
@($CacheDir, $UvCacheDir, $TmpDir) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
}
$env:UV_CACHE_DIR = $UvCacheDir
$env:TMP = $TmpDir
$env:TEMP = $TmpDir
# ==========================================

$EnvName = "face_env"
$PythonVersion = "3.12"
$UvVersion = "0.9.14"
$UvZipUrl = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvExePath = Join-Path $ScriptPath "uv.exe"
$PythonExePath = Join-Path $ScriptPath "$EnvName\Scripts\python.exe"
$WheelsDir = Join-Path $ScriptPath "wheels"

function Write-Status {
    param([string]$Message, [string]$Type = "INFO")
    $prefix = switch ($Type) {
        "INFO" { " [INFO]  " } "WARN" { " [WARN]  " }
        "ERROR" { " [ERROR] " } "SUCCESS" { " [OK]    " } default { " [INFO]  " }
    }
    $color = switch ($Type) {
        "INFO" { "Cyan" } "WARN" { "Yellow" } "ERROR" { "Red" }
        "SUCCESS" { "Green" } default { "White" }
    }
    Write-Host "$prefix$Message" -ForegroundColor $color
}

function Write-Step {
    param([string]$Message, [int]$Step, [int]$Total)
    Write-Host ""
    Write-Host ">>> Stage [$Step/$Total]: $Message" -ForegroundColor Magenta
}

function Invoke-UvPipInstall {
    param([string]$Command)
    Write-Host "   > uv pip install $Command" -ForegroundColor DarkGray
    $process = Start-Process -FilePath $UvExePath `
        -ArgumentList "pip install --python `"$PythonExePath`" $Command" `
        -NoNewWindow -Wait -PassThru
    return $process.ExitCode
}

function Test-PythonModule {
    param([string]$Module)
    $code = "import importlib.util as u; import sys; sys.exit(0 if u.find_spec('$Module') else 1)"
    & $PythonExePath -c $code 2>$null
    return ($LASTEXITCODE -eq 0)
}

Write-Host ""
Write-Host " ==========================================" -ForegroundColor Green
Write-Host "   AI_FaceRecognitor - Portable Installer" -ForegroundColor Yellow
Write-Host "   python $PythonVersion + torch cu128 + insightface + onnxruntime-gpu" -ForegroundColor Gray
Write-Host " ==========================================" -ForegroundColor Green
Write-Host ""

# === 0. uv ===
Write-Step "Checking uv package manager (idempotent)..." 0 5
if (Test-Path $UvExePath) {
    Write-Status "uv.exe already in project folder - skipping." "SUCCESS"
} else {
    $sysUv = Get-Command uv -ErrorAction SilentlyContinue
    if ($sysUv) {
        $v = & uv --version
        Write-Status "System uv found: $v. Copying into project (portable)." "INFO"
        Copy-Item $sysUv.Source $UvExePath -Force
    } else {
        Write-Status "Downloading uv $UvVersion..." "INFO"
        $uvZip = Join-Path $ScriptPath "uv.zip"
        Invoke-WebRequest -Uri $UvZipUrl -OutFile $uvZip -ErrorAction Stop
        $uvTmp = Join-Path $ScriptPath "uv_tmp"
        if (Test-Path $uvTmp) { Remove-Item $uvTmp -Recurse -Force }
        Expand-Archive -Path $uvZip -DestinationPath $uvTmp -Force
        $extracted = Get-ChildItem $uvTmp -Directory | Select-Object -First 1
        $srcDir = if ($extracted) { $extracted.FullName } else { $uvTmp }
        Copy-Item (Join-Path $srcDir "uv.exe") $UvExePath -Force
        Copy-Item (Join-Path $srcDir "uvx.exe") (Join-Path $ScriptPath "uvx.exe") -Force -ErrorAction SilentlyContinue
        Remove-Item $uvTmp -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $uvZip -Force -ErrorAction SilentlyContinue
    }
    Write-Status "uv ready." "SUCCESS"
}

# === 1. venv ===
Write-Step "Checking Python environment ($EnvName)..." 1 5
$venvOk = (Test-Path $PythonExePath) -and (Test-Path (Join-Path $ScriptPath "$EnvName\pyvenv.cfg"))
if ($venvOk) {
    Write-Status "$EnvName already exists - skipping creation." "SUCCESS"
} else {
    if (Test-Path $EnvName) {
        Write-Status "'$EnvName' exists but is broken - recreating." "WARN"
        Remove-Item -Path $EnvName -Recurse -Force
    }
    & $UvExePath venv $EnvName --python $PythonVersion
    if ($LASTEXITCODE -ne 0) { Write-Status "Failed to create environment." "ERROR"; exit 1 }
    # compatibility entry point: face_env\python.exe -> Scripts\python.exe
    $pyTarget = Join-Path $ScriptPath "$EnvName\Scripts\python.exe"
    $pyLink = Join-Path $ScriptPath "$EnvName\python.exe"
    if (-not (Test-Path $pyLink)) {
        try {
            $null = New-Item -ItemType HardLink -Path $pyLink -Target $pyTarget -ErrorAction Stop
        } catch {
            Set-Content -Path $pyLink -Value "@`"%~dp0Scripts\python.exe`" %*" -Encoding ASCII
        }
    }
    Write-Status "Environment created (Python $PythonVersion via uv)." "SUCCESS"
}

# === 2. PyTorch cu128 (DLL provider for onnxruntime; proven Trellis2 stack) ===
Write-Step "Checking PyTorch 2.8.0+cu128..." 2 5
if (Test-PythonModule "torch") {
    Write-Status "torch already installed - skipping." "SUCCESS"
} else {
    $rc = Invoke-UvPipInstall "torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
    if ($rc -ne 0) { Write-Status "torch install failed." "ERROR"; exit 1 }
    Write-Status "torch installed. CUDA 12 DLLs will be reused from torch\lib." "SUCCESS"
}

# === 3. Face recognition stack (each package checked individually) ===
Write-Step "Checking insightface + onnxruntime-gpu 1.22.x + deps..." 3 5
$pkgs = @(
    @{ name = "insightface";        module = "insightface" },
    @{ name = "onnxruntime-gpu==1.22.0"; module = "onnxruntime" },
    @{ name = "opencv-python";      module = "cv2" },
    @{ name = "tqdm";               module = "tqdm" },
    @{ name = "numpy";              module = "numpy" },
    # insightface 2.1 использует устаревший skimage API (Transform.estimate):
    # в scikit-image >= 0.26 это FutureWarning, в 2.2 будет поломка - пин ниже
    @{ name = '"scikit-image<0.26"'; module = "skimage" },
    @{ name = "pycurl";             module = "curl" }   # vendored wheel below
)
foreach ($p in $pkgs) {
    if (Test-PythonModule $p.module) {
        Write-Status "$($p.name) already installed - skipping." "SUCCESS"
        continue
    }
    # pycurl: prefer the vendored proven wheel (cp312), pip fallback otherwise
    if ($p.name -eq "pycurl") {
        $wheel = Get-ChildItem $WheelsDir -Filter "pycurl*cp312*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($wheel) {
            $rc = Invoke-UvPipInstall "`"$($wheel.FullName)`""
        } else {
            Write-Status "pycurl wheel not found in wheels\ - will use urllib fallback." "WARN"
            continue
        }
    } else {
        $rc = Invoke-UvPipInstall $p.name
    }
    if ($rc -ne 0) { Write-Status "Failed: $($p.name)" "ERROR"; exit 1 }
}

# === 4. InsightFace models (src/download_models.py is idempotent) ===
Write-Step "Checking insightface models buffalo_l (pycurl download if missing)..." 4 5
& $PythonExePath (Join-Path $ScriptPath "src/download_models.py")
if ($LASTEXITCODE -ne 0) {
    Write-Status "Model download failed - run src/download_models.py later to retry." "ERROR"
    exit 1
}

# === 5. Verify (cheap: imports only, no model load) ===
Write-Step "Verifying installation..." 5 5
$verify = @'
import sys
import importlib.util as u
mods = ["torch", "insightface", "onnxruntime", "cv2", "tqdm", "numpy", "curl"]
missing = [m for m in mods if u.find_spec(m) is None]
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(1)
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
import onnxruntime as ort
print("onnxruntime:", ort.__version__, "| providers:", ort.get_available_providers())
print("ALL OK")
'@
$verifyFile = Join-Path $env:TEMP "verify_fr.py"
Set-Content -Path $verifyFile -Value $verify -Encoding UTF8
& $PythonExePath $verifyFile
if ($LASTEXITCODE -ne 0) { Write-Status "Verification failed." "ERROR"; exit 1 }
Remove-Item $verifyFile -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host " ==========================================" -ForegroundColor Green
Write-Host "   INSTALLATION COMPLETE (or already up to date)" -ForegroundColor Green
Write-Host "   Run:  powershell -File .\run_faces.ps1 --help" -ForegroundColor Yellow
Write-Host " ==========================================" -ForegroundColor Green
