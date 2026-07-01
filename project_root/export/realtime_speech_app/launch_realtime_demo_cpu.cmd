@echo off
setlocal

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
set "CHECKPOINT=%PROJECT_ROOT%\checkpoints\train_balanced_res_20260614_091134\masknet_best.pth"
set "RECORD_DIR=%PROJECT_ROOT%\results\realtime_recordings"
set "SNAPSHOT_DIR=%PROJECT_ROOT%\results\realtime_snapshots"

set "PYTHON_EXE=python"
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    "%PROJECT_ROOT%\.venv\Scripts\python.exe" --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
)

mkdir "%RECORD_DIR%" >nul 2>nul
mkdir "%SNAPSHOT_DIR%" >nul 2>nul

"%PYTHON_EXE%" -m src.tools.realtime_demo ^
    --method masknet ^
    --checkpoint "%CHECKPOINT%" ^
    --device cpu ^
    --visualize ^
    --stats-interval 2 ^
    --record-dir "%RECORD_DIR%" ^
    --snapshot-dir "%SNAPSHOT_DIR%"

pause
