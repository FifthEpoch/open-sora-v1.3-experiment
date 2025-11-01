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

# compute PSNR/SSIM/LPIPS
pip install scikit-image lpips