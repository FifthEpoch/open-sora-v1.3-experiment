# login node (no GPU)
module purge
conda create -y -n opensora13 python=3.9
conda activate opensora13

python -m pip install -U pip setuptools wheel

# Open-Sora v1.3 tested pins (CUDA 12.1 wheels)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2

pip install --index-url https://download.pytorch.org/whl/cu121 \
  xformers==0.0.25.post1

# pull the v1.3-pinned packages (no CUDA build yet)
pip install -r requirements/requirements-cu121.txt

# Install base requirements
pip install -r requirements/requirements.txt

# Install eval requirements (includes decord, lpips, scikit-image)
pip install -r requirements/requirements-eval.txt

# Install VAE requirements (includes opencv-python, pillow, einops)
pip install -r requirements/requirements-vae.txt

# Install huggingface-hub for checkpoint download script
pip install huggingface-hub