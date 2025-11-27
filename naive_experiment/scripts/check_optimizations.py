#!/usr/bin/env python3
"""
Check what optimizations are available in the environment
"""
import sys

print("=" * 60)
print("OPTIMIZATION AVAILABILITY CHECK")
print("=" * 60)

# Check flash-attn
try:
    import flash_attn
    print("✓ flash-attn:", flash_attn.__version__)
    flash_attn_available = True
except ImportError as e:
    print("✗ flash-attn: NOT AVAILABLE")
    print(f"  Error: {e}")
    flash_attn_available = False

# Check apex
try:
    import apex
    from apex.normalization import FusedLayerNorm
    print("✓ apex: AVAILABLE (with FusedLayerNorm)")
    apex_available = True
except ImportError as e:
    print("✗ apex: NOT AVAILABLE")
    print(f"  Error: {e}")
    apex_available = False

# Check xformers
try:
    import xformers
    print("✓ xformers:", xformers.__version__)
except ImportError as e:
    print("✗ xformers: NOT AVAILABLE")
    print(f"  Error: {e}")

# Check torch
import torch
print("\n" + "=" * 60)
print("PYTORCH INFO")
print("=" * 60)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU compute capability:", torch.cuda.get_device_capability(0))

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if flash_attn_available and apex_available:
    print("✓ All optimizations available!")
    print("  Use: enable_flash_attn=True, enable_layernorm_kernel=True")
elif flash_attn_available:
    print("⚠ flash-attn available but apex missing")
    print("  Use: enable_flash_attn=True, enable_layernorm_kernel=False")
elif apex_available:
    print("⚠ apex available but flash-attn missing")
    print("  Use: enable_flash_attn=False, enable_layernorm_kernel=True")
else:
    print("⚠ No optimizations available (will be 2-3x slower)")
    print("  Use: enable_flash_attn=False, enable_layernorm_kernel=False")
    print("\nThis is OK for correctness, but for better performance:")
    print("  - Install flash-attn: pip install flash-attn")
    print("  - Install apex: pip install apex (or build from source)")

print("=" * 60)

