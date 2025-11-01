# On a login node
conda activate opensora13
python - <<'PY'
import torch, xformers, pkgutil
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
import opensora
print("Open-Sora import OK")
for lib in ["flash_attn", "apex", "decord", "av", "cv2", "lpips", "skimage"]:
    print(lib, "OK" if pkgutil.find_loader(lib) else "MISSING")
PY