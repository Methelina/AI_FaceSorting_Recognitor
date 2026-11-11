@echo off
rem ============================================================
rem  AI_FaceRecognitor - Runtime launcher (mirror of run_faces.ps1)
rem  All arguments are passed to find_faces.py as-is, e.g.:
rem    run_faces.bat --search-dir O:\pics --target-dir O:\pics\person --samples "a.jpg,b.jpg"
rem    run_faces.bat --help
rem ============================================================
setlocal
chcp 65001 >nul
title AI_FaceRecognitor
set "SCRIPT_DIR=%~dp0"

if not exist "%SCRIPT_DIR%face_env\Scripts\python.exe" (
    echo [ERROR] Environment not found. Run FaceRecognitor_portable.bat first.
    echo.
    pause
    exit /b 1
)

rem isolation: caches inside project, UTF-8 io, GPU device 0
set "UV_CACHE_DIR=%SCRIPT_DIR%.cache\uv"
set "HF_HOME=%SCRIPT_DIR%.cache\huggingface"
set "TMP=%SCRIPT_DIR%.cache\tmp"
set "TEMP=%TMP%"
set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONIOENCODING=utf-8"
if not exist "%TMP%" mkdir "%TMP%" >nul 2>&1

"%SCRIPT_DIR%face_env\Scripts\python.exe" "%SCRIPT_DIR%find_faces.py" %*

set "EC=%ERRORLEVEL%"
echo.
echo Press any key to exit...
pause >nul
exit /b %EC%
