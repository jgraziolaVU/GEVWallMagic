╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                        ◈  PARALLAX STUDIO  ◈                                 ║
║                                                                              ║
║            Transform photos into depth-enhanced video wall content           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝


QUICK START
═══════════════════════════════════════════════════════════════════════════════

  1. Make sure you have an NVIDIA GPU (RTX 3000/4000/5000 series recommended)

  2. Double-click "install.bat" and follow the prompts
     (This takes 10-20 minutes and only needs to be done once)

  3. Double-click "Parallax Studio" shortcut on your Desktop
     (Or double-click "run.bat")

  4. The app opens in your web browser - start creating!


WHAT'S IN THIS FOLDER
═══════════════════════════════════════════════════════════════════════════════

  install.bat          One-click installer for all dependencies
  run.bat              Launcher for the application  
  parallax_studio.py   The main application code
  requirements.txt     Python package list (for manual installation)
  README.txt           This file

  ⚠️  IMPORTANT: Keep all these files together in the same folder!
      The launcher (run.bat) expects parallax_studio.py to be right
      next to it. Moving files individually will break the app.


SYSTEM REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

  REQUIRED:
  • Windows 10 or 11 (64-bit)
  • NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better recommended)
  • 16GB RAM minimum (32GB recommended for 4K output)
  • 5GB free disk space for installation
  • Internet connection (for initial setup)

  WILL BE INSTALLED AUTOMATICALLY:
  • Miniconda (Python environment manager)
  • Python 3.11
  • PyTorch with CUDA
  • Apple SHARP
  • Streamlit
  • FFmpeg


MANUAL INSTALLATION (Advanced Users)
═══════════════════════════════════════════════════════════════════════════════

  If you prefer to install manually or the automatic installer fails:

  1. Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

  2. Open Anaconda Prompt and run:

     conda create -n parallax_studio python=3.11
     conda activate parallax_studio
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     
     git clone https://github.com/apple/ml-sharp.git
     cd ml-sharp
     pip install -r requirements.txt
     pip install -e .
     cd ..
     
     pip install streamlit huggingface-hub Pillow plyfile tqdm
     conda install -c conda-forge ffmpeg

  3. Download the model checkpoint:
     
     huggingface-cli download --include sharp_2572gikvuh.pt --local-dir . apple/Sharp

  4. Run the app:
     
     streamlit run parallax_studio.py


TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

  "parallax_studio.py not found"
  ──────────────────────────────
  • Make sure parallax_studio.py is in the same folder as run.bat
  • Don't move or rename any of the application files
  • Re-download the complete application folder if files are missing

  "NVIDIA GPU not detected"
  ─────────────────────────
  • Install the latest NVIDIA drivers from: https://www.nvidia.com/drivers
  • Restart your computer after installing drivers
  • Make sure your monitor is plugged into the GPU, not motherboard

  "Conda is not recognized"
  ─────────────────────────
  • Restart your computer after installing Miniconda
  • Make sure you checked "Add to PATH" during Miniconda installation
  • Try running install.bat again

  "Out of memory" during rendering
  ────────────────────────────────
  • Use "Half" or "Preview" resolution in Step 4
  • Close other GPU-intensive applications
  • Reduce the number of frames (shorter duration)

  "FFmpeg not found"
  ──────────────────
  • Download FFmpeg from: https://ffmpeg.org/download.html
  • Add FFmpeg to your system PATH
  • Or reinstall using: conda install -c conda-forge ffmpeg

  App won't open in browser
  ─────────────────────────
  • Manually open: http://localhost:8501
  • Check if another application is using port 8501
  • Try: streamlit run parallax_studio.py --server.port=8502

  Rendering fails or produces black frames
  ────────────────────────────────────────
  • Make sure CUDA is working: python -c "import torch; print(torch.cuda.is_available())"
  • Update NVIDIA drivers
  • Try with a smaller/simpler image first


TIPS FOR BEST RESULTS
═══════════════════════════════════════════════════════════════════════════════

  ✓ Use images with clear depth layers
    (foreground, middle-ground, background)

  ✓ Landscapes, cityscapes, and interior shots work great

  ✓ Higher resolution input = better quality output

  ✓ Start with subtle amplitude (0.10-0.15) and increase if needed

  ✓ For video walls, 32:9 at 5120×1440 is standard

  ✗ Avoid flat subjects (documents, walls, close-up faces)

  ✗ Avoid images with lots of transparency or reflections


GETTING HELP
═══════════════════════════════════════════════════════════════════════════════

  SHARP Documentation:
  https://github.com/apple/ml-sharp

  SHARP Research Paper:
  https://arxiv.org/abs/2512.10685

  Streamlit Documentation:
  https://docs.streamlit.io


LICENSE
═══════════════════════════════════════════════════════════════════════════════

  This application uses Apple SHARP which is subject to Apple's license terms.
  See the SHARP repository for full license details.


═══════════════════════════════════════════════════════════════════════════════
                           ◈ Enjoy creating! ◈
═══════════════════════════════════════════════════════════════════════════════
