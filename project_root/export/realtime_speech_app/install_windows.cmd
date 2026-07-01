@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
    echo During setup, enable "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

if not exist ".venv" (
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements_realtime.txt

echo.
echo Install complete.
echo Start the app with launch_realtime_demo.cmd.
pause
