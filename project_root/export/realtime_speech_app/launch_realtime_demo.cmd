@echo off
setlocal

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
set "CHECKPOINT=%PROJECT_ROOT%\checkpoints\train_balanced_res_20260614_091134\masknet_best.pth"
set "RECORD_DIR=%PROJECT_ROOT%\results\realtime_recordings"
set "SNAPSHOT_DIR=%PROJECT_ROOT%\results\realtime_snapshots"
set "LOCK_DIR=%TEMP%\realtime_speech_enhancement_demo.lock"

if exist "%LOCK_DIR%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*src.tools.realtime_demo*' }) { exit 0 } else { exit 1 }" >nul 2>nul
    if errorlevel 1 (
        rmdir "%LOCK_DIR%" >nul 2>nul
    )
)

mkdir "%LOCK_DIR%" >nul 2>nul
if errorlevel 1 (
    echo Realtime Speech Enhancement Monitor is already running.
    echo Close the existing window before starting another instance.
    echo.
    pause
    exit /b 0
)

if not exist "%CHECKPOINT%" (
    echo Checkpoint not found:
    echo   %CHECKPOINT%
    echo.
    rmdir "%LOCK_DIR%" >nul 2>nul
    pause
    exit /b 1
)

set "PYTHON_EXE=python"
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    "%PROJECT_ROOT%\.venv\Scripts\python.exe" --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
)

mkdir "%RECORD_DIR%" >nul 2>nul
mkdir "%SNAPSHOT_DIR%" >nul 2>nul

echo Starting Realtime Speech Enhancement Monitor...
echo Project: %PROJECT_ROOT%
echo Python:  %PYTHON_EXE%
echo.
echo Close the UI window or press Ctrl+C here to stop.
echo.

"%PYTHON_EXE%" -m src.tools.realtime_demo ^
    --method masknet ^
    --checkpoint "%CHECKPOINT%" ^
    --visualize ^
    --stats-interval 2 ^
    --record-dir "%RECORD_DIR%" ^
    --snapshot-dir "%SNAPSHOT_DIR%"

rmdir "%LOCK_DIR%" >nul 2>nul

echo.
echo Demo stopped.
pause
