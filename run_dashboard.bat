@echo off
REM Run HVAC dashboard. Prefers conda env python-hvac; falls back to pip.
cd /d "%~dp0"

where conda >nul 2>&1
if %errorlevel% equ 0 (
  call conda activate python-hvac 2>nul
  if %errorlevel% equ 0 (
    echo Starting HVAC dashboard at http://127.0.0.1:8000
    echo Press Ctrl+C to stop.
    python -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8000
    pause
    exit /b 0
  )
  echo Conda env "python-hvac" not found. Create it with:
  echo   conda env create -f environment.yml
  echo Then run this script again.
  echo.
)

REM No conda or env missing: try current Python + pip
echo Conda not found or env missing. Using current Python.
python -c "import uvicorn, fastapi, jinja2" 2>nul
if %errorlevel% neq 0 (
  echo Installing dashboard dependencies (--user to avoid Access denied)...
  python -m pip install --user -e . >nul 2>&1
  python -m pip install --user -r requirements-dashboard.txt
  if %errorlevel% neq 0 (
    echo.
    echo Install failed. Try manually:
    echo   python -m pip install --user -e .
    echo   python -m pip install --user -r requirements-dashboard.txt
    pause
    exit /b 1
  )
)

python -c "import hvac" 2>nul
if %errorlevel% neq 0 (
  echo Installing hvac package in development mode...
  python -m pip install --user -e .
)

echo.
echo Starting HVAC dashboard at http://127.0.0.1:8000
echo Press Ctrl+C to stop.
python -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8000
pause
