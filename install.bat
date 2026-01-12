@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: PARALLAX STUDIO INSTALLER
:: ============================================================================
:: This script installs everything needed to run Parallax Studio.
:: Just double-click this file and follow the prompts.
:: ============================================================================

title Parallax Studio - Installation

:: Colors and formatting via PowerShell
cls
echo.
echo  ======================================================
echo.
echo        ◈  PARALLAX STUDIO INSTALLER  ◈
echo.
echo  ======================================================
echo.
echo  This will install all required software:
echo.
echo    • Python environment (via Miniconda)
echo    • Apple SHARP (3D Gaussian Splatting)
echo    • Streamlit (Web Interface)
echo    • FFmpeg (Video Processing)
echo.
echo  Estimated time: 10-20 minutes
echo  Disk space required: ~5 GB
echo.
echo  ======================================================
echo.

:: Check for administrator privileges (needed for some installations)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo  [!] NOTE: Running without administrator privileges.
    echo      Some features may require you to run as Administrator.
    echo.
)

pause
cls

:: ============================================================================
:: STEP 1: Check for NVIDIA GPU
:: ============================================================================

echo.
echo  [Step 1/7] Checking for NVIDIA GPU...
echo.

nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    echo  [WARNING] NVIDIA GPU not detected or drivers not installed.
    echo.
    echo  Parallax Studio requires an NVIDIA GPU with CUDA support
    echo  for rendering. The app will install, but rendering will fail.
    echo.
    echo  Please install NVIDIA drivers from:
    echo  https://www.nvidia.com/drivers
    echo.
    pause
) else (
    echo  [OK] NVIDIA GPU detected!
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo.
)

:: ============================================================================
:: STEP 2: Check/Install Miniconda
:: ============================================================================

echo.
echo  [Step 2/7] Checking for Conda...
echo.

where conda >nul 2>&1
if %errorLevel% neq 0 (
    echo  Conda not found. Installing Miniconda...
    echo.
    
    :: Download Miniconda
    echo  Downloading Miniconda installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%TEMP%\miniconda_installer.exe'"
    
    if not exist "%TEMP%\miniconda_installer.exe" (
        echo  [ERROR] Failed to download Miniconda.
        echo  Please download manually from: https://docs.conda.io/en/latest/miniconda.html
        pause
        exit /b 1
    )
    
    echo  Running Miniconda installer...
    echo  Please follow the installation prompts.
    echo  IMPORTANT: Check "Add to PATH" when prompted!
    echo.
    
    start /wait "" "%TEMP%\miniconda_installer.exe"
    
    del "%TEMP%\miniconda_installer.exe"
    
    echo.
    echo  [!] Miniconda installed. Please RESTART this installer
    echo      to continue with the remaining steps.
    echo.
    pause
    exit /b 0
) else (
    echo  [OK] Conda is installed!
    where conda
    echo.
)

:: ============================================================================
:: STEP 3: Create Conda Environment
:: ============================================================================

echo.
echo  [Step 3/7] Creating Python environment...
echo.

:: Check if environment already exists
call conda env list | findstr /C:"parallax_studio" >nul
if %errorLevel% equ 0 (
    echo  Environment 'parallax_studio' already exists.
    set /p RECREATE="  Recreate it? (y/n): "
    if /i "!RECREATE!"=="y" (
        echo  Removing existing environment...
        call conda env remove -n parallax_studio -y
    ) else (
        echo  Keeping existing environment.
        goto :skip_env_create
    )
)

echo  Creating new environment with Python 3.11...
call conda create -n parallax_studio python=3.11 -y

if %errorLevel% neq 0 (
    echo  [ERROR] Failed to create conda environment.
    pause
    exit /b 1
)

:skip_env_create
echo  [OK] Environment ready!
echo.

:: ============================================================================
:: STEP 4: Install PyTorch with CUDA
:: ============================================================================

echo.
echo  [Step 4/7] Installing PyTorch with CUDA support...
echo.

call conda activate parallax_studio

:: Install PyTorch with CUDA 12.1 (works with most modern GPUs)
echo  This may take several minutes...
call pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if %errorLevel% neq 0 (
    echo  [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

:: Verify CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo.
echo  [OK] PyTorch installed!
echo.

:: ============================================================================
:: STEP 5: Install SHARP
:: ============================================================================

echo.
echo  [Step 5/7] Installing Apple SHARP...
echo.

:: Create app directory
set "APP_DIR=%USERPROFILE%\ParallaxStudio"
if not exist "%APP_DIR%" mkdir "%APP_DIR%"
cd /d "%APP_DIR%"

:: Clone SHARP repository if not present
if not exist "%APP_DIR%\ml-sharp" (
    echo  Cloning SHARP repository...
    git clone https://github.com/apple/ml-sharp.git
    
    if %errorLevel% neq 0 (
        echo.
        echo  [ERROR] Git clone failed. 
        echo  Please install Git from: https://git-scm.com/download/win
        echo  Then restart this installer.
        pause
        exit /b 1
    )
) else (
    echo  SHARP repository already exists, updating...
    cd ml-sharp
    git pull
    cd ..
)

:: Install SHARP requirements
echo  Installing SHARP dependencies...
cd ml-sharp
call pip install -r requirements.txt
call pip install -e .
cd ..

if %errorLevel% neq 0 (
    echo  [ERROR] Failed to install SHARP.
    pause
    exit /b 1
)

echo  [OK] SHARP installed!
echo.

:: ============================================================================
:: STEP 6: Install Additional Dependencies
:: ============================================================================

echo.
echo  [Step 6/7] Installing additional dependencies...
echo.

call pip install streamlit>=1.31.0
call pip install huggingface-hub>=0.20.0
call pip install Pillow>=10.0.0
call pip install plyfile
call pip install tqdm

:: Download SHARP model checkpoint
echo.
echo  Downloading SHARP model checkpoint (~500MB)...
echo  This may take a few minutes depending on your connection...
echo.

python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='apple/Sharp', filename='sharp_2572gikvuh.pt', local_dir='%USERPROFILE%\.cache\sharp')"

if %errorLevel% neq 0 (
    echo  [WARNING] Model download failed. Will retry on first app launch.
) else (
    echo  [OK] Model checkpoint downloaded!
)

echo.

:: ============================================================================
:: STEP 7: Install FFmpeg
:: ============================================================================

echo.
echo  [Step 7/7] Checking for FFmpeg...
echo.

where ffmpeg >nul 2>&1
if %errorLevel% neq 0 (
    echo  FFmpeg not found. Installing via conda...
    call conda install -c conda-forge ffmpeg -y
    
    if %errorLevel% neq 0 (
        echo.
        echo  [WARNING] FFmpeg installation failed.
        echo  Please install manually from: https://ffmpeg.org/download.html
        echo  Video export will not work without FFmpeg.
    ) else (
        echo  [OK] FFmpeg installed!
    )
) else (
    echo  [OK] FFmpeg is already installed!
    where ffmpeg
)

echo.

:: ============================================================================
:: STEP 8: Create App Files
:: ============================================================================

echo.
echo  Setting up application files...
echo.

:: Copy the Streamlit app (user should place parallax_studio.py in same folder as install.bat)
if exist "%~dp0parallax_studio.py" (
    copy "%~dp0parallax_studio.py" "%APP_DIR%\parallax_studio.py" >nul
    echo  [OK] App file copied!
) else (
    echo  [NOTE] parallax_studio.py not found in installer directory.
    echo         Please copy it manually to: %APP_DIR%
)

:: Create the run.bat in app directory
echo @echo off > "%APP_DIR%\run.bat"
echo title Parallax Studio >> "%APP_DIR%\run.bat"
echo call conda activate parallax_studio >> "%APP_DIR%\run.bat"
echo cd /d "%APP_DIR%" >> "%APP_DIR%\run.bat"
echo echo. >> "%APP_DIR%\run.bat"
echo echo  Starting Parallax Studio... >> "%APP_DIR%\run.bat"
echo echo  The app will open in your web browser. >> "%APP_DIR%\run.bat"
echo echo. >> "%APP_DIR%\run.bat"
echo echo  To stop the app, close this window or press Ctrl+C >> "%APP_DIR%\run.bat"
echo echo. >> "%APP_DIR%\run.bat"
echo streamlit run parallax_studio.py --server.maxUploadSize=500 >> "%APP_DIR%\run.bat"
echo pause >> "%APP_DIR%\run.bat"

echo  [OK] Launcher created!
echo.

:: Create desktop shortcut
echo  Creating desktop shortcut...
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\Parallax Studio.lnk'); $s.TargetPath = '%APP_DIR%\run.bat'; $s.WorkingDirectory = '%APP_DIR%'; $s.IconLocation = 'shell32.dll,13'; $s.Save()"
echo  [OK] Desktop shortcut created!
echo.

:: ============================================================================
:: INSTALLATION COMPLETE
:: ============================================================================

cls
echo.
echo  ======================================================
echo.
echo        ◈  INSTALLATION COMPLETE!  ◈
echo.
echo  ======================================================
echo.
echo  Parallax Studio has been installed to:
echo  %APP_DIR%
echo.
echo  To run the application:
echo.
echo    • Double-click "Parallax Studio" on your Desktop
echo.
echo    OR
echo.
echo    • Double-click "run.bat" in the app folder
echo.
echo  ======================================================
echo.
echo  The app will open in your default web browser.
echo  First launch may take a minute to initialize.
echo.
echo  ======================================================
echo.

set /p LAUNCH="  Launch Parallax Studio now? (y/n): "
if /i "%LAUNCH%"=="y" (
    echo.
    echo  Starting Parallax Studio...
    start "" "%APP_DIR%\run.bat"
)

echo.
echo  Thank you for installing Parallax Studio!
echo.
pause
exit /b 0
