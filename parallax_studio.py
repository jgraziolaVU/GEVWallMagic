"""
SHARP Parallax Studio
A premium interface for creating depth-enhanced video wall content

Run with: streamlit run parallax_studio.py
"""

import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
import time
import subprocess
import shutil
import os
from PIL import Image
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SHARP Parallax Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "checkpoint_name": "sharp_2572gikvuh.pt",
    "checkpoint_dir": Path.home() / ".cache" / "sharp",
    "hf_repo": "apple/Sharp",
}

# Aspect ratio presets with common resolutions
ASPECT_RATIOS = {
    "16:9 (Standard Widescreen)": {
        "ratio": 16/9,
        "resolutions": [
            ("4K UHD", 3840, 2160),
            ("2K QHD", 2560, 1440),
            ("1080p Full HD", 1920, 1080),
            ("720p HD", 1280, 720),
        ]
    },
    "16:10 (Common Monitor)": {
        "ratio": 16/10,
        "resolutions": [
            ("WQXGA", 2560, 1600),
            ("WUXGA", 1920, 1200),
            ("WXGA+", 1680, 1050),
            ("WXGA", 1280, 800),
        ]
    },
    "21:9 (Ultrawide)": {
        "ratio": 21/9,
        "resolutions": [
            ("UWQHD", 3440, 1440),
            ("UW-UXGA", 2560, 1080),
            ("UW-HD", 2560, 1080),
        ]
    },
    "32:9 (Super Ultrawide)": {
        "ratio": 32/9,
        "resolutions": [
            ("Dual 4K", 7680, 2160),
            ("Dual QHD", 5120, 1440),
            ("Dual FHD", 3840, 1080),
        ]
    },
    "Custom": {
        "ratio": None,
        "resolutions": []
    }
}

# ============================================================================
# PREMIUM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a25;
        --bg-card: linear-gradient(145deg, #151520 0%, #0d0d14 100%);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #a855f7;
        --accent-glow: rgba(99, 102, 241, 0.4);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-accent: rgba(99, 102, 241, 0.3);
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    .stApp {
        background: var(--bg-primary);
        background-image: 
            radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem 4rem;
    }
    
    .hero-container {
        text-align: center;
        padding: 3rem 0 4rem;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-accent);
        border-radius: 100px;
        padding: 0.5rem 1.25rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--accent-primary);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 50%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 1rem 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .studio-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .studio-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.4) 50%, transparent 100%);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .card-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 32px var(--accent-glow);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .card-description {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0;
    }
    
    .step-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-right: 0.75rem;
    }
    
    .stSlider > div > div {
        background: var(--bg-tertiary) !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 24px var(--accent-glow) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px var(--accent-glow) !important;
    }
    
    .stProgress > div > div {
        background: var(--bg-tertiary) !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary)) !important;
        border-radius: 10px !important;
    }
    
    .metric-container {
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid var(--border-subtle);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-ready {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-processing {
        background: rgba(99, 102, 241, 0.15);
        color: var(--accent-primary);
        border: 1px solid var(--border-accent);
    }
    
    .status-pending {
        background: rgba(100, 116, 139, 0.15);
        color: var(--text-muted);
        border: 1px solid var(--border-subtle);
    }
    
    .preview-container {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid var(--border-subtle);
    }
    
    .preview-container img, .preview-container video {
        border-radius: 8px;
        width: 100%;
    }
    
    .param-group {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .param-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .param-hint {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    .cli-output {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #c9d1d9;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }
    
    .success-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .gpu-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: var(--success);
    }
    
    .studio-footer {
        text-align: center;
        padding: 3rem 0 1rem;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    
    .footer-logo {
        font-weight: 600;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
    }
    
    .streamlit-expanderContent h3 {
        color: var(--text-primary) !important;
        font-size: 1.3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .streamlit-expanderContent h4 {
        color: var(--accent-primary) !important;
        font-size: 1.1rem !important;
        margin-top: 1.5rem !important;
    }
    
    .streamlit-expanderContent p, .streamlit-expanderContent li {
        color: var(--text-secondary) !important;
        line-height: 1.7 !important;
    }
    
    .streamlit-expanderContent strong {
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent code {
        background: var(--bg-tertiary) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        color: var(--accent-primary) !important;
    }
    
    .streamlit-expanderContent hr {
        border-color: var(--border-subtle) !important;
        margin: 1.5rem 0 !important;
    }
    
    .streamlit-expanderContent a {
        color: var(--accent-primary) !important;
    }
    
    h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
    p, span, label { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if 'source_image' not in st.session_state:
    st.session_state.source_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'gaussian_path' not in st.session_state:
    st.session_state.gaussian_path = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'work_dir' not in st.session_state:
    st.session_state.work_dir = None
if 'cli_output' not in st.session_state:
    st.session_state.cli_output = ""
if 'target_width' not in st.session_state:
    st.session_state.target_width = 5120
if 'target_height' not in st.session_state:
    st.session_state.target_height = 1440
if 'target_ratio' not in st.session_state:
    st.session_state.target_ratio = 32/9


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_status_badge(status):
    icons = {'ready': '✓', 'processing': '◌', 'pending': '○'}
    return f'<span class="status-badge status-{status}">{icons[status]} {status.title()}</span>'


def check_gpu():
    """Check for CUDA GPU availability."""
    try:
        result = subprocess.run(
            ["python", "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() == "True"
    except:
        return False


def get_gpu_name():
    """Get GPU name if available."""
    try:
        result = subprocess.run(
            ["python", "-c", "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except:
        return "Unknown"


def ensure_checkpoint():
    """Download SHARP checkpoint if not present."""
    checkpoint_path = CONFIG["checkpoint_dir"] / CONFIG["checkpoint_name"]
    
    if checkpoint_path.exists():
        return checkpoint_path
    
    CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    
    # Download from HuggingFace
    cmd = [
        "huggingface-cli", "download",
        "--include", CONFIG["checkpoint_name"],
        "--local-dir", str(CONFIG["checkpoint_dir"]),
        CONFIG["hf_repo"]
    ]
    
    subprocess.run(cmd, check=True)
    return checkpoint_path


def process_image_to_aspect(image, target_ratio, target_width, target_height):
    """Crop and resize image to target aspect ratio and resolution."""
    width, height = image.size
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    else:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def run_sharp_predict(input_dir, output_dir, checkpoint_path):
    """Run SHARP prediction to generate gaussian splat."""
    cmd = [
        "sharp", "predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-c", str(checkpoint_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def run_sharp_render(gaussian_dir, output_dir, checkpoint_path):
    """Run SHARP render to generate video from gaussians."""
    cmd = [
        "sharp", "render",
        "-i", str(gaussian_dir),
        "-o", str(output_dir),
        "-c", str(checkpoint_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def create_oscillation_video(gaussian_path, output_path, params):
    """
    Create custom oscillation video using gsplat.
    This renders the gaussian splat with an oscillating camera path.
    """
    render_script = f'''
import numpy as np
import torch
from pathlib import Path
from plyfile import PlyData
import subprocess

def load_ply(path):
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    return {{
        "xyz": np.stack([v["x"], v["y"], v["z"]], axis=-1),
        "scales": np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1),
        "rots": np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1),
        "sh": np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1),
        "opacity": v["opacity"]
    }}

def render_oscillation(ply_path, out_dir, width, height, frames, amplitude, fps):
    from gsplat import rasterization
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_ply(ply_path)
    device = "cuda"
    
    means = torch.tensor(data["xyz"], dtype=torch.float32, device=device)
    scales = torch.exp(torch.tensor(data["scales"], dtype=torch.float32, device=device))
    quats = torch.tensor(data["rots"], dtype=torch.float32, device=device)
    colors = torch.tensor(data["sh"], dtype=torch.float32, device=device)
    opacities = torch.sigmoid(torch.tensor(data["opacity"], dtype=torch.float32, device=device))
    
    fx = fy = width * 0.8
    cx, cy = width / 2, height / 2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)
    
    # Scene center (SHARP convention: scene at +z)
    scene_z = float(means[:, 2].mean())
    
    for i in range(frames):
        t = 2 * np.pi * i / frames
        x_offset = amplitude * np.sin(t)
        
        # Camera position
        cam_pos = np.array([x_offset, 0.0, scene_z - 2.5])
        
        # Look-at rotation (OpenCV: x-right, y-down, z-forward)
        R = np.eye(3)
        T = -R @ cam_pos
        
        viewmat = np.eye(4)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = T
        viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device)
        
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
        )
        
        frame = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(frame).save(out_dir / f"frame_{{i:05d}}.png")
        
        if (i + 1) % 30 == 0:
            print(f"Rendered {{i + 1}}/{{frames}}")
    
    # Assemble video with ffmpeg
    video_path = out_dir.parent / "parallax_loop.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(out_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ], check=True)
    
    return str(video_path)

if __name__ == "__main__":
    result = render_oscillation(
        "{gaussian_path}",
        "{output_path}",
        {params['width']},
        {params['height']},
        {params['frames']},
        {params['amplitude']},
        {params['fps']}
    )
    print(f"VIDEO_OUTPUT:{{result}}")
'''
    
    # Write and execute render script
    script_path = Path(output_path).parent / "render_script.py"
    with open(script_path, "w") as f:
        f.write(render_script)
    
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True, text=True
    )
    
    # Extract video path from output
    for line in result.stdout.split("\n"):
        if line.startswith("VIDEO_OUTPUT:"):
            return True, line.replace("VIDEO_OUTPUT:", ""), result.stdout + result.stderr
    
    return False, None, result.stdout + result.stderr


def setup_work_directory():
    """Create and return working directory structure."""
    if st.session_state.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="parallax_studio_"))
        (work_dir / "input").mkdir()
        (work_dir / "gaussians").mkdir()
        (work_dir / "frames").mkdir()
        (work_dir / "output").mkdir()
        st.session_state.work_dir = work_dir
    return st.session_state.work_dir


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">◈ Powered by Apple SHARP + 3D Gaussian Splatting</div>
    <h1 class="hero-title">Parallax Studio</h1>
    <p class="hero-subtitle">Transform static images into living, breathing displays with photorealistic depth parallax for your video wall</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# COLLAPSIBLE INFO SECTIONS
# ============================================================================

with st.expander("📖 How to Use This App", expanded=False):
    st.markdown("""
    ### Welcome to Parallax Studio!
    
    This app transforms a regular photograph into a video with **depth and dimension** — making it look like you're peering through a window into the scene rather than looking at a flat image.
    
    ---
    
    #### What You'll Need
    - **A high-resolution image** — The better the quality, the better the result. Photos from GoPro, DSLR, or modern smartphones work great.
    - **A few minutes** — Processing takes 1-3 minutes depending on your settings.
    
    ---
    
    #### Step-by-Step Walkthrough
    
    **Step 1: Upload Your Image**
    Click the upload area or drag-and-drop your photo. Supported formats include PNG, JPG, and TIFF.
    
    **Step 2: Choose Format & Resolution**
    Select the aspect ratio that matches your display:
    - **16:9** — Standard TVs and monitors
    - **16:10** — Many desktop monitors
    - **21:9** — Ultrawide monitors
    - **32:9** — Super ultrawide or video walls
    
    The app will intelligently crop your image to fit, centering on the middle of the photo.
    
    **Step 3: Generate 3D Structure**
    Click "Run SHARP" and wait. The AI analyzes your photo and figures out what's close, what's far, and everything in between. This creates an invisible 3D model of your scene.
    
    **Step 4: Render the Parallax Video**
    Adjust the settings to your taste:
    - **Amplitude** — How dramatic the depth effect appears. Start with 0.15 (subtle) and increase if you want more "wow factor."
    - **Duration** — How long the loop lasts before repeating. 10 seconds is a good starting point.
    - **Frame Rate** — 30fps is smooth; 60fps is silky but takes longer to render.
    
    Click "Begin Render" and watch the progress bar.
    
    **Step 5: Download & Enjoy**
    Preview your video right in the browser, then download the MP4 file. Copy it to your video wall system and set it to loop!
    
    ---
    
    #### Tips for Best Results
    
    ✓ **Scenes with clear depth work best** — A landscape with foreground flowers, middle-ground trees, and distant mountains will look amazing.
    
    ✓ **Avoid flat scenes** — A photo of a wall or document won't have much parallax to show.
    
    ✓ **Higher resolution = better quality** — The AI has more information to work with.
    
    ✓ **Start with subtle settings** — You can always re-render with more dramatic amplitude if the effect is too tame.
    """)

with st.expander("ℹ️ About Parallax Studio", expanded=False):
    st.markdown("""
    ### About This Application
    
    **Parallax Studio** was built to bring museum-quality depth effects to standard LED video walls without requiring expensive specialized hardware like lenticular displays or head-tracking cameras.
    
    ---
    
    #### The Technology
    
    This app combines two cutting-edge technologies:
    
    **Apple SHARP** (Sharp Monocular View Synthesis)
    A neural network developed by Apple's machine learning research team that can look at a single photograph and understand its 3D structure. In less than a second, it creates a complete 3D representation of your scene using a technique called "3D Gaussian Splatting."
    
    **3D Gaussian Splatting (3DGS)**
    A revolutionary method for representing 3D scenes as millions of tiny, colored, semi-transparent ellipsoids. Unlike traditional 3D models made of triangles, Gaussian splats can be rendered extremely quickly while maintaining photorealistic quality.
    
    ---
    
    #### How the Parallax Effect Works
    
    When you look at the real world and move your head, nearby objects shift more than distant objects. This is called **motion parallax**, and it's one of the most powerful depth cues your brain uses.
    
    The rendered video simulates a camera slowly drifting left and right. Because SHARP has figured out the 3D structure of your scene, objects at different depths move at different speeds — just like reality. Your brain interprets this as genuine depth, even though you're watching a flat screen.
    
    ---
    
    #### Credits & References
    
    **SHARP Paper:**
    "Sharp Monocular View Synthesis in Less Than a Second"
    Lars Mescheder, Wei Dong, Shiwei Li, et al.
    arXiv:2512.10685 (2025)
    
    **Built With:**
    - [Streamlit](https://streamlit.io) — User interface
    - [PyTorch](https://pytorch.org) — Deep learning framework
    - [gsplat](https://github.com/nerfstudio-project/gsplat) — Gaussian splatting renderer
    - [FFmpeg](https://ffmpeg.org) — Video encoding
    
    ---
    
    #### System Requirements
    
    - **NVIDIA GPU with CUDA** — Required for rendering (RTX 3000/4000/5000 series recommended)
    - **16GB+ RAM** — For processing high-resolution images
    - **Python 3.10+** — Runtime environment
    """)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# GPU STATUS BAR
# ============================================================================

gpu_available = check_gpu()
gpu_name = get_gpu_name() if gpu_available else "Not detected"

col_gpu1, col_gpu2, col_gpu3 = st.columns([1, 2, 1])
with col_gpu2:
    if gpu_available:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="gpu-indicator">⚡ GPU Active: {gpu_name}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="status-badge status-pending">⚠ No CUDA GPU detected - rendering will fail</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# STATUS OVERVIEW
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    status = 'ready' if st.session_state.source_image else 'pending'
    st.markdown(f"""
    <div class="metric-container">
        {get_status_badge(status)}
        <div class="metric-label" style="margin-top: 0.75rem;">Source Image</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status = 'ready' if st.session_state.processed_image else 'pending'
    st.markdown(f"""
    <div class="metric-container">
        {get_status_badge(status)}
        <div class="metric-label" style="margin-top: 0.75rem;">32:9 Processed</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    status = 'ready' if st.session_state.gaussian_path else 'pending'
    st.markdown(f"""
    <div class="metric-container">
        {get_status_badge(status)}
        <div class="metric-label" style="margin-top: 0.75rem;">3D Gaussian</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    status = 'ready' if st.session_state.video_path else 'pending'
    st.markdown(f"""
    <div class="metric-container">
        {get_status_badge(status)}
        <div class="metric-label" style="margin-top: 0.75rem;">Parallax Video</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# STEP 1: IMAGE UPLOAD
# ============================================================================

st.markdown("""
<div class="studio-card">
    <div class="card-header">
        <div class="card-icon">📷</div>
        <div>
            <h3 class="card-title"><span class="step-indicator">1</span>Source Image</h3>
            <p class="card-description">Upload a high-resolution frame from your GoPro footage</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.source_image = image
        
        st.markdown('<div class="preview-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with upload_col2:
    if st.session_state.source_image:
        img = st.session_state.source_image
        st.markdown(f"""
        <div class="param-group">
            <div class="param-label">Image Properties</div>
            <br>
            <strong style="color: #f1f5f9;">Resolution</strong>
            <p style="font-family: 'JetBrains Mono'; font-size: 1.1rem;">{img.size[0]} × {img.size[1]}</p>
            <strong style="color: #f1f5f9;">Aspect Ratio</strong>
            <p style="font-family: 'JetBrains Mono'; font-size: 1.1rem;">{img.size[0]/img.size[1]:.2f}:1</p>
            <strong style="color: #f1f5f9;">Target</strong>
            <p style="font-family: 'JetBrains Mono'; font-size: 1.1rem;">5120 × 1440 (32:9)</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# STEP 2: FORMAT SELECTION & PROCESSING
# ============================================================================

if st.session_state.source_image:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">⬡</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">2</span>Format for Your Display</h3>
                <p class="card-description">Choose aspect ratio and resolution, then crop and scale your image</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    format_col1, format_col2 = st.columns([1, 1])
    
    with format_col1:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        st.markdown('<div class="param-label">Aspect Ratio</div>', unsafe_allow_html=True)
        
        aspect_choice = st.selectbox(
            "Select aspect ratio",
            options=list(ASPECT_RATIOS.keys()),
            index=3,  # Default to 32:9
            label_visibility="collapsed",
            help="Choose the aspect ratio that matches your display"
        )
        
        # Show description for each aspect ratio
        ratio_descriptions = {
            "16:9 (Standard Widescreen)": "Standard HDTVs, most monitors, YouTube videos",
            "16:10 (Common Monitor)": "MacBooks, many desktop monitors, productivity displays",
            "21:9 (Ultrawide)": "Ultrawide gaming monitors, cinematic displays",
            "32:9 (Super Ultrawide)": "Samsung Odyssey, video walls, dual-monitor replacement",
            "Custom": "Enter your own dimensions"
        }
        st.markdown(f'<p class="param-hint">{ratio_descriptions[aspect_choice]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with format_col2:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        st.markdown('<div class="param-label">Resolution</div>', unsafe_allow_html=True)
        
        if aspect_choice == "Custom":
            custom_col1, custom_col2 = st.columns(2)
            with custom_col1:
                custom_width = st.number_input("Width", min_value=640, max_value=8192, value=1920, step=1)
            with custom_col2:
                custom_height = st.number_input("Height", min_value=360, max_value=4320, value=1080, step=1)
            target_width, target_height = custom_width, custom_height
            target_ratio = custom_width / custom_height
        else:
            resolution_options = ASPECT_RATIOS[aspect_choice]["resolutions"]
            resolution_labels = [f"{name} ({w}×{h})" for name, w, h in resolution_options]
            
            res_choice = st.selectbox(
                "Select resolution",
                options=resolution_labels,
                index=0,
                label_visibility="collapsed",
                help="Higher resolution = better quality but longer processing"
            )
            
            # Extract dimensions from choice
            res_idx = resolution_labels.index(res_choice)
            _, target_width, target_height = resolution_options[res_idx]
            target_ratio = ASPECT_RATIOS[aspect_choice]["ratio"]
        
        total_pixels = target_width * target_height
        mp = total_pixels / 1_000_000
        st.markdown(f'<p class="param-hint">{mp:.1f} megapixels • {target_width}×{target_height}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Preview crop area
    proc_col1, proc_col2 = st.columns([3, 1])
    
    with proc_col2:
        if st.button("Process Image", key="process_btn", use_container_width=True):
            with st.spinner("Processing..."):
                processed = process_image_to_aspect(
                    st.session_state.source_image,
                    target_ratio,
                    target_width,
                    target_height
                )
                st.session_state.processed_image = processed
                st.session_state.target_width = target_width
                st.session_state.target_height = target_height
                st.session_state.target_ratio = target_ratio
                st.rerun()
    
    with proc_col1:
        if st.session_state.processed_image:
            st.markdown('<div class="preview-container">', unsafe_allow_html=True)
            st.image(st.session_state.processed_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            w, h = st.session_state.processed_image.size
            ratio_str = f"{w/h:.2f}:1"
            st.markdown(f"""
            <p style="text-align: center; margin-top: 0.5rem; font-size: 0.85rem;">
                ✓ Processed to {w} × {h} ({ratio_str})
            </p>
            """, unsafe_allow_html=True)


# ============================================================================
# STEP 3: GENERATE 3D GAUSSIAN
# ============================================================================

if st.session_state.processed_image:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">◇</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">3</span>Generate 3D Structure</h3>
                <p class="card-description">Run SHARP to extract depth and create 3D Gaussian splat</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    gauss_col1, gauss_col2 = st.columns([3, 1])
    
    with gauss_col2:
        if st.button("Run SHARP", key="sharp_btn", use_container_width=True):
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Setup directories
            work_dir = setup_work_directory()
            input_path = work_dir / "input" / "frame.png"
            st.session_state.processed_image.save(input_path)
            
            # Ensure checkpoint
            status_placeholder.markdown("*Checking SHARP checkpoint...*")
            try:
                checkpoint_path = ensure_checkpoint()
                status_placeholder.markdown(f"*Checkpoint: {checkpoint_path.name}*")
            except Exception as e:
                st.error(f"Failed to download checkpoint: {e}")
                st.stop()
            
            # Run SHARP
            status_placeholder.markdown("*Running SHARP prediction...*")
            progress_placeholder.progress(0.3)
            
            success, output = run_sharp_predict(
                work_dir / "input",
                work_dir / "gaussians",
                checkpoint_path
            )
            
            progress_placeholder.progress(1.0)
            st.session_state.cli_output = output
            
            if success:
                # Find the generated PLY file
                ply_files = list((work_dir / "gaussians").glob("*.ply"))
                if ply_files:
                    st.session_state.gaussian_path = ply_files[0]
                    status_placeholder.markdown("*✓ 3D Gaussian splat generated*")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("SHARP completed but no PLY file found")
            else:
                st.error("SHARP prediction failed")
                with st.expander("CLI Output"):
                    st.code(output)
    
    with gauss_col1:
        if st.session_state.gaussian_path:
            file_size = st.session_state.gaussian_path.stat().st_size / (1024 * 1024)
            st.markdown(f"""
            <div class="param-group" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">◇</div>
                <div class="param-label">3D Gaussian Splat Generated</div>
                <p style="font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
                    {st.session_state.gaussian_path.name}<br>
                    {file_size:.1f} MB
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.cli_output:
                with st.expander("SHARP Output Log"):
                    st.markdown(f'<div class="cli-output">{st.session_state.cli_output}</div>', unsafe_allow_html=True)


# ============================================================================
# STEP 4: RENDER PARALLAX VIDEO
# ============================================================================

if st.session_state.gaussian_path:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">▶</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">4</span>Render Parallax Video</h3>
                <p class="card-description">Configure camera motion and render the final looping video</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    render_col1, render_col2 = st.columns([1, 1])
    
    with render_col1:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        st.markdown('<div class="param-label">Camera Motion</div>', unsafe_allow_html=True)
        
        amplitude = st.slider(
            "Oscillation Amplitude",
            min_value=0.05,
            max_value=0.40,
            value=0.15,
            step=0.01,
            help="How far the virtual camera moves side-to-side"
        )
        st.markdown('<p class="param-hint">Lower = subtle depth, Higher = dramatic parallax</p>', unsafe_allow_html=True)
        
        duration = st.slider(
            "Loop Duration (seconds)",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Length of the seamless loop"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with render_col2:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        st.markdown('<div class="param-label">Output Settings</div>', unsafe_allow_html=True)
        
        fps = st.selectbox(
            "Frame Rate",
            options=[24, 30, 60],
            index=1,
            help="Higher = smoother but longer render"
        )
        
        # Resolution options based on processed image
        full_w, full_h = st.session_state.target_width, st.session_state.target_height
        half_w, half_h = full_w // 2, full_h // 2
        preview_w, preview_h = full_w // 4, full_h // 4
        
        resolution = st.selectbox(
            "Resolution",
            options=[
                f"Full ({full_w}×{full_h})",
                f"Half ({half_w}×{half_h})",
                f"Preview ({preview_w}×{preview_h})"
            ],
            index=0,
            help="Lower for faster test renders"
        )
        
        if "Full" in resolution:
            width, height = full_w, full_h
        elif "Half" in resolution:
            width, height = half_w, half_h
        else:
            width, height = preview_w, preview_h
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Render info
    total_frames = duration * fps
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.9rem;">
        Total frames: <strong>{total_frames}</strong> • 
        Estimated time: <strong>{total_frames * 0.1:.0f}–{total_frames * 0.3:.0f} seconds</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Render button
    render_btn_col1, render_btn_col2, render_btn_col3 = st.columns([1, 2, 1])
    with render_btn_col2:
        if st.button("✦  Begin Render", key="render_btn", use_container_width=True):
            if not gpu_available:
                st.error("CUDA GPU required for rendering")
                st.stop()
            
            work_dir = st.session_state.work_dir
            
            params = {
                "width": width,
                "height": height,
                "frames": total_frames,
                "amplitude": amplitude,
                "fps": fps
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("*Initializing gsplat renderer...*")
            
            success, video_path, output = create_oscillation_video(
                str(st.session_state.gaussian_path),
                str(work_dir / "frames"),
                params
            )
            
            progress_bar.progress(1.0)
            
            if success and video_path:
                st.session_state.video_path = Path(video_path)
                status_text.markdown("*✓ Render complete*")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Rendering failed")
                with st.expander("Render Output"):
                    st.code(output)


# ============================================================================
# STEP 5: PREVIEW & DOWNLOAD
# ============================================================================

if st.session_state.video_path and st.session_state.video_path.exists():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">✦</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">5</span>Preview & Export</h3>
                <p class="card-description">Your parallax video is ready for the video wall</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-banner">
        <span style="font-size: 2rem;">✓</span>
        <h4 style="color: #10b981 !important; margin: 0.5rem 0;">Render Complete</h4>
        <p style="color: #94a3b8; margin: 0;">Your parallax video has been generated successfully</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Video preview
    st.markdown('<div class="preview-container">', unsafe_allow_html=True)
    st.video(str(st.session_state.video_path))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File info
    video_size = st.session_state.video_path.stat().st_size / (1024 * 1024)
    st.markdown(f"""
    <p style="text-align: center; margin-top: 0.5rem; font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #64748b;">
        {st.session_state.video_path.name} • {video_size:.1f} MB
    </p>
    """, unsafe_allow_html=True)
    
    # Download buttons
    st.markdown("<br>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    
    with dl_col1:
        with open(st.session_state.video_path, "rb") as f:
            st.download_button(
                "⬇ Download MP4",
                data=f.read(),
                file_name="parallax_loop.mp4",
                mime="video/mp4",
                use_container_width=True
            )
    
    with dl_col2:
        with open(st.session_state.gaussian_path, "rb") as f:
            st.download_button(
                "⬇ Download PLY",
                data=f.read(),
                file_name="gaussian_splat.ply",
                mime="application/octet-stream",
                use_container_width=True
            )
    
    with dl_col3:
        if st.button("↺ Start Over", use_container_width=True):
            # Clean up temp directory
            if st.session_state.work_dir:
                shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
            
            for key in ['source_image', 'processed_image', 'gaussian_path', 'video_path', 
                        'work_dir', 'cli_output', 'target_width', 'target_height', 'target_ratio']:
                if key in ['target_width']:
                    st.session_state[key] = 5120
                elif key in ['target_height']:
                    st.session_state[key] = 1440
                elif key in ['target_ratio']:
                    st.session_state[key] = 32/9
                else:
                    st.session_state[key] = None
            st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="studio-footer">
    <span class="footer-logo">◈ Parallax Studio</span>
    <br>
    <span>Powered by Apple SHARP & 3D Gaussian Splatting</span>
    <br>
    <span style="font-size: 0.75rem; opacity: 0.7;">Made for video walls, ultrawide monitors, and immersive displays</span>
</div>
""", unsafe_allow_html=True)
