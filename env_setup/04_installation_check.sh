# On a login node
conda activate opensora13
python - <<'PY'
import torch, xformers, pkgutil
print("="*60)
print("INSTALLATION CHECK")
print("="*60)
print("\nTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))

# Check Open-Sora
try:
    import opensora
    print("\n✓ Open-Sora import OK")
except Exception as e:
    print(f"\n✗ Open-Sora import FAILED: {e}")

# Check essential libraries
print("\nChecking dependencies:")
for lib in ["flash_attn", "apex", "decord", "av", "cv2", "lpips", "skimage", "einops"]:
    status = "✓ OK" if pkgutil.find_loader(lib) else "✗ MISSING"
    print(f"  {lib:15s} {status}")

# Check additional experiment dependencies
print("\nChecking experiment-specific dependencies:")
for lib in ["pandas", "numpy", "tqdm", "huggingface_hub", "colossalai", "mmengine"]:
    status = "✓ OK" if pkgutil.find_loader(lib) else "✗ MISSING"
    print(f"  {lib:20s} {status}")

print("\n" + "="*60)
print("If all checks pass, you're ready to run the experiment!")
print("="*60)
PY