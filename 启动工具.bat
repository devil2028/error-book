@echo off
cd /d "%~dp0"

REM 首次运行自动创建虚拟环境并安装依赖
if not exist ".venv\Scripts\python.exe" (
    echo [1/2] First run: creating virtual environment ...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv. Please make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    echo [2/2] Installing dependencies ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo Starting ... browser will open automatically.
".venv\Scripts\python.exe" app.py
pause
