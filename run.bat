@echo off
setlocal

:: ============================================================================
:: PARALLAX STUDIO LAUNCHER
:: ============================================================================
:: Double-click this file to start Parallax Studio.
:: The app will open in your default web browser.
:: ============================================================================

title ◈ Parallax Studio

:: Set the app directory (same folder as this batch file)
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

cls
echo.
echo  ======================================================
echo.
echo        ◈  PARALLAX STUDIO  ◈
echo.
echo  ======================================================
echo.

:: ============================================================================
:: Check if conda environment exists
:: ============================================================================

call conda --version >nul 2>&1
if %errorLevel% neq 0 (
    echo  [ERROR] Conda is not installed or not in PATH.
    echo.
    echo  Please run install.bat first, or install Miniconda from:
    echo  https://docs.conda.io/en/latest/miniconda.html
    echo.
    goto :error_exit
)

:: ============================================================================
:: Activate environment
:: ============================================================================

echo  Activating Python environment...
call conda activate parallax_studio

if %errorLevel% neq 0 (
    echo.
    echo  [ERROR] Could not activate 'parallax_studio' environment.
    echo.
    echo  Please run install.bat first to set up the environment.
    echo.
    goto :error_exit
)

echo  [OK] Environment activated
echo.

:: ============================================================================
:: Check for app file
:: ============================================================================

if not exist "%APP_DIR%parallax_studio.py" (
    echo  [ERROR] parallax_studio.py not found in:
    echo  %APP_DIR%
    echo.
    echo  Please ensure the app file is in the same folder as this launcher.
    echo.
    goto :error_exit
)

:: ============================================================================
:: Check GPU status
:: ============================================================================

echo  Checking GPU status...
python -c "import torch; gpu = torch.cuda.is_available(); name = torch.cuda.get_device_name(0) if gpu else 'None'; print(f'  GPU: {name}' if gpu else '  [WARNING] No CUDA GPU detected')"
echo.

:: ============================================================================
:: Launch Streamlit
:: ============================================================================

echo  Starting Parallax Studio...
echo.
echo  ======================================================
echo.
echo   The app will open in your web browser automatically.
echo.
echo   If it doesn't open, go to: http://localhost:8501
echo.
echo   To STOP the app: Close this window or press Ctrl+C
echo.
echo  ======================================================
echo.

:: Start Streamlit with optimized settings
:: --server.maxUploadSize=500   Allow large image uploads (500MB)
:: --server.headless=false      Open browser automatically
:: --browser.gatherUsageStats=false  Disable telemetry

streamlit run parallax_studio.py ^
    --server.maxUploadSize=500 ^
    --server.headless=false ^
    --browser.gatherUsageStats=false

:: If we get here, Streamlit has stopped
echo.
echo  Parallax Studio has stopped.
echo.
pause
exit /b 0

:: ============================================================================
:: Error handler
:: ============================================================================

:error_exit
echo.
echo  ======================================================
echo  Press any key to exit...
echo  ======================================================
pause >nul
exit /b 1
