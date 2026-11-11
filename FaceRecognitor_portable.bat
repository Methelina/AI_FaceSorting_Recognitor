@echo off
rem ============================================================
rem  AI_FaceRecognitor - Portable Installer (TRUE CMD, no PowerShell)
rem  Modelled on FaceRecognitor_portable.ps1 / Trellis2_portable.ps1
rem
rem  Fully IDEMPOTENT: safe to run repeatedly, finished stages skip.
rem  Fully SELF-CONTAINED: caches, venv and models live inside the
rem  project folder. ComfyUI is NOT used or touched. Models download
rem  from official insightface release via pycurl (download_models.py).
rem
rem  Requires only: cmd.exe + internet (uv downloads Python itself)
rem ============================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
title AI_FaceRecognitor Portable Installer
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "ENV_NAME=face_env"
set "PYTHON_VERSION=3.12"
set "UV_VERSION=0.9.14"
set "UV_EXE=%SCRIPT_DIR%uv.exe"
set "PYTHON_EXE=%SCRIPT_DIR%%ENV_NAME%\Scripts\python.exe"
set "WHEELS_DIR=%SCRIPT_DIR%wheels"

rem --- PORTABILITY ISOLATION: caches inside the project folder ---------
set "UV_CACHE_DIR=%SCRIPT_DIR%.cache\uv"
set "TMP=%SCRIPT_DIR%.cache\tmp"
set "TEMP=%TMP%"
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%" >nul 2>&1
if not exist "%TMP%" mkdir "%TMP%" >nul 2>&1

echo.
echo  ==========================================
echo    AI_FaceRecognitor - Portable Installer
echo    python %PYTHON_VERSION% + torch cu128 + insightface + onnxruntime-gpu
echo  ==========================================
echo.

rem === 0. uv ============================================================
echo [Stage 0/5] uv package manager (idempotent)
if exist "%UV_EXE%" (
    echo  [OK]    uv.exe already in project folder - skipping.
    goto :stage1
)
where uv >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%v in ('uv --version') do echo  [INFO]  System uv found: %%v
    echo  [INFO]  Copying into project (portable).
    for /f "tokens=*" %%p in ('where uv') do copy /y "%%p" "%UV_EXE%" >nul & goto :uv_ok
)
echo  [INFO]  Downloading uv %UV_VERSION%...
set "UV_ZIP=%TMP%\uv.zip"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    echo  [ERROR] Unsupported architecture: ARM64
    goto :fail
)
set "UV_URL=https://releases.astral.sh/github/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
where curl >nul 2>&1
if not errorlevel 1 (
    curl -L --connect-timeout 30 -o "%UV_ZIP%" "%UV_URL%"
    if errorlevel 1 goto :uv_dl_fail
) else (
    rem no curl: use certutil-free bitsadmin as last resort
    bitsadmin /transfer uvdl "https://releases.astral.sh/github/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip" "%UV_ZIP%" >nul 2>&1
    if not exist "%UV_ZIP%" goto :uv_dl_fail
)
:uv_dl_ok
if not exist "%SCRIPT_DIR%uv_tmp" mkdir "%SCRIPT_DIR%uv_tmp"
tar -xf "%UV_ZIP%" -C "%SCRIPT_DIR%uv_tmp"
if exist "%UV_EXE%" del "%UV_EXE%"
rem zip contains uv.exe (maybe inside a subfolder) - find it:
for /r "%SCRIPT_DIR%uv_tmp" %%f in (uv.exe) do copy /y "%%f" "%UV_EXE%" >nul & goto :uv_extracted
:uv_extracted
del "%UV_ZIP%" >nul 2>&1
rmdir /s /q "%SCRIPT_DIR%uv_tmp" >nul 2>&1
:uv_ok
if not exist "%UV_EXE%" (
    echo  [ERROR] uv.exe not found after download/extract.
    goto :fail
)
echo  [OK]    uv ready.

rem === 1. venv ==========================================================
:stage1
echo.
echo [Stage 1/5] Python environment (%ENV_NAME%) (idempotent)
if exist "%PYTHON_EXE%" if exist "%SCRIPT_DIR%%ENV_NAME%\pyvenv.cfg" (
    echo  [OK]    %ENV_NAME% already exists - skipping creation.
    goto :stage2
)
if exist "%SCRIPT_DIR%%ENV_NAME%" (
    echo  [WARN]  '%ENV_NAME%' exists but is broken - recreating.
    rmdir /s /q "%SCRIPT_DIR%%ENV_NAME%"
)
"%UV_EXE%" venv "%ENV_NAME%" --python %PYTHON_VERSION%
if errorlevel 1 (
    echo  [ERROR] Failed to create environment.
    goto :fail
)
rem compatibility entry point: face_env\python.exe -> Scripts\python.exe
if not exist "%SCRIPT_DIR%%ENV_NAME%\python.exe" (
    copy /y "%PYTHON_EXE%" "%SCRIPT_DIR%%ENV_NAME%\python.exe" >nul 2>&1
)
echo  [OK]    Environment created (Python %PYTHON_VERSION% via uv).

rem === 2. PyTorch cu128 =================================================
:stage2
echo.
echo [Stage 2/5] PyTorch 2.8.0+cu128 (DLL provider for onnxruntime)
"%PYTHON_EXE%" -c "import torch" >nul 2>&1
if not errorlevel 1 (
    echo  [OK]    torch already installed - skipping.
    goto :stage3
)
"%UV_EXE%" pip install --python "%PYTHON_EXE%" torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo  [ERROR] torch install failed.
    goto :fail
)
echo  [OK]    torch installed. CUDA 12 DLLs will be reused from torch\lib.

rem === 3. Face recognition stack ========================================
:stage3
echo.
echo [Stage 3/5] insightface + onnxruntime-gpu 1.22.x + deps

call :mod_install insightface insightface
if errorlevel 1 goto :fail
call :mod_install onnxruntime onnxruntime onnxruntime-gpu==1.22.0
if errorlevel 1 goto :fail
call :mod_install cv2 cv2 opencv-python
if errorlevel 1 goto :fail
call :mod_install tqdm tqdm tqdm
if errorlevel 1 goto :fail
call :mod_install numpy numpy numpy
if errorlevel 1 goto :fail
rem insightface 2.1 uses deprecated skimage Transform.estimate API:
rem FutureWarning on scikit-image >= 0.26, hard break on 2.2 - pin below
call :mod_install skimage skimage "scikit-image<0.26"
if errorlevel 1 goto :fail
rem pycurl: prefer vendored proven wheel (cp312); if missing, urllib fallback
"%PYTHON_EXE%" -c "import curl" >nul 2>&1
if not errorlevel 1 (
    echo  [OK]    pycurl already installed - skipping.
    goto :stage4
)
set "PYCURL_WHEEL="
for %%f in ("%WHEELS_DIR%\pycurl*cp312*.whl") do set "PYCURL_WHEEL=%%f"
if defined PYCURL_WHEEL (
    "%UV_EXE%" pip install --python "%PYTHON_EXE%" "!PYCURL_WHEEL!"
    if errorlevel 1 goto :fail
    echo  [OK]    pycurl installed from vendored wheel.
) else (
    echo  [WARN]  pycurl wheel not found in wheels\ - downloader will use urllib fallback.
)

rem === 4. Models ========================================================
:stage4
echo.
echo [Stage 4/5] insightface models buffalo_l (download if missing)
"%PYTHON_EXE%" "%SCRIPT_DIR%download_models.py"
if errorlevel 1 (
    echo  [ERROR] Model download failed - run download_models.py later to retry.
    goto :fail
)

rem === 5. Verify ========================================================
:stage5
echo.
echo [Stage 5/5] Verifying installation...
"%PYTHON_EXE%" -c "import torch, insightface, onnxruntime, cv2, tqdm, numpy, curl; import torch as t; print('torch:', t.__version__, '| cuda:', t.cuda.is_available()); import onnxruntime as o; print('onnxruntime:', o.__version__, '| providers:', o.get_available_providers()); print('ALL OK')"
if errorlevel 1 (
    echo  [ERROR] Verification failed.
    goto :fail
)

echo.
echo  ==========================================
echo    INSTALLATION COMPLETE (or already up to date)
echo    Run:  run_faces.bat --help
echo  ==========================================
echo.
pause
exit /b 0

:mod_install
rem %1=import name  %2=check module  %3=optional pip spec (default %1)
rem returns errorlevel 1 on install failure; sets MOD_INSTALLED=skipped/ok
set "PIP_SPEC=%~3"
if not defined PIP_SPEC set "PIP_SPEC=%~1"
"%PYTHON_EXE%" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('%~2') else 1)" >nul 2>&1
if not errorlevel 1 (
    echo  [OK]    %~1 already installed - skipping.
    exit /b 0
)
"%UV_EXE%" pip install --python "%PYTHON_EXE%" %PIP_SPEC%
if errorlevel 1 (
    echo  [ERROR] Failed: %~1
    exit /b 1
)
echo  [OK]    %~1 installed.
exit /b 0

:uv_dl_fail
echo  [ERROR] Download failed. Check internet connection.
goto :fail

:fail
echo.
echo  [ERROR] Installation aborted.
echo.
pause
exit /b 1
