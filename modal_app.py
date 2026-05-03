import os
import sys
import subprocess
from pathlib import Path
import modal

# 1. Define the Modal App
app = modal.App("cot-mech-interp")

# 2. Define the dependencies (from requirements.txt)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "transformers>=4.41.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        "transformer-lens>=1.17.0",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "statsmodels>=0.14.0",
        "pandas>=2.2.0",
        "pyarrow>=16.0.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.9.0",
        "editdistance>=0.6.3",
        "jsonlines>=4.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "click>=8.1.0",
        "pytest>=8.0.0",
        "pytest-cov>=5.0.0"
    )
    # debian_slim has no git — install it first so pip can clone from GitHub
    .run_commands("apt-get update && apt-get install -y git")
    # Install circuit-tracer from source as specified in requirements.txt
    .pip_install("git+https://github.com/decoderesearch/circuit-tracer.git")
    # Modal 1.x: Add local files directly to the image
    .add_local_dir("src", remote_path="/root/cot-mech-interp/src")
    .add_local_dir("scripts", remote_path="/root/cot-mech-interp/scripts")
    .add_local_file("config.yaml", remote_path="/root/cot-mech-interp/config.yaml")
)

# 3. Persistent Volume for data and artifacts
storage_vol = modal.Volume.from_name("cot-interp-storage", create_if_missing=True)

# 4. Remote function to execute any script
@app.function(
    image=image,
    gpu="L40S",  # 48GB VRAM is strictly required because Phase 3 (Logit Attributions) OOMs on 24GB
    volumes={"/mnt/storage": storage_vol},
    secrets=[
        # This will securely pass your HF_TOKEN to the container so you can download Gemma
        modal.Secret.from_name("Hugging-face-secret", required_keys=["HF_TOKEN"])
    ],
    timeout=86400,  # Allow up to 24 hours of execution time
)
def run_script_remote(script_path: str):
    # Set working directory to our project
    os.chdir("/root/cot-mech-interp")
    
    # Ensure the storage directories exist in the Volume
    os.makedirs("/mnt/storage/data", exist_ok=True)
    os.makedirs("/mnt/storage/artifacts", exist_ok=True)
    
    # Create symlinks so the scripts find 'data' and 'artifacts' in the current directory
    # but the actual files are safely on the Modal Volume
    if not os.path.exists("data"):
        os.symlink("/mnt/storage/data", "data")
    if not os.path.exists("artifacts"):
        os.symlink("/mnt/storage/artifacts", "artifacts")
        
    print(f"--- Starting execution of {script_path} ---")
    
    # Run the script using the same Python executable in the container
    cmd = [sys.executable, script_path, "--config", "config.yaml"]
    
    # Stream stdout/stderr in real-time so logs appear in the Modal dashboard
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    if result.returncode != 0:
        print(f"--- Error: Script {script_path} failed with return code {result.returncode} ---")
        sys.exit(result.returncode)
    else:
        print(f"--- Success: Script {script_path} completed successfully ---")


# 6. Local Entrypoint
@app.local_entrypoint()
def main(script: str = "scripts/03b_turpin_prerun.py"):
    """
    Usage: modal run modal_app.py --script scripts/04_generate_graphs.py
    """
    print(f"Deploying to Modal and executing: {script}")
    run_script_remote.remote(script)
