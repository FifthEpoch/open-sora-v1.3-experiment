# Aspect Ratio Confusion - Portrait vs Landscape

## Problem Discovered

Both `test_conditioning_output.mp4` and `test_strong_conditioning_output.mp4` are **portrait orientation** (560×744), but UCF-101 videos are **landscape** (640×480)!

## Root Cause: H:W vs W:H Notation

### Traditional Aspect Ratio Notation (W:H)
- **4:3** = 4 units wide : 3 units tall = **LANDSCAPE** (like old TVs, UCF-101)
- **16:9** = 16 units wide : 9 units tall = **LANDSCAPE** (modern widescreen)
- **9:16** = 9 units wide : 16 units tall = **PORTRAIT** (TikTok, phone videos)

### Open-Sora's Notation (H:W) ⚠️
Open-Sora uses **H:W** notation (inverted from traditional):

```python
# opensora/datasets/aspect.py line 26:
# H:W
ASPECT_RATIO_MAP = {
    "9:16": "0.56",  # H:W = 9/16 = 0.56 → PORTRAIT (taller than wide)
    "3:4": "0.75",   # H:W = 3/4 = 0.75 → LANDSCAPE (wider than tall)
    "4:3": "1.33",   # H:W = 4/3 = 1.33 → PORTRAIT (taller than wide)
    "16:9": "1.78",  # H:W = 16/9 = 1.78 → PORTRAIT (taller than wide)
}
```

## What Happened

Our config specified:
```python
aspect_ratio = "4:3"  # Comment says "640x480" (landscape)
```

But Open-Sora interpreted this as:
- **H:W = 4:3** → H/W = 1.33
- H=740, W=555 → **740 tall × 555 wide** = **PORTRAIT**!
- Final video (after VAE rounding): **744 tall × 560 wide** = **PORTRAIT**

## The Confusion

| Traditional (W:H) | Open-Sora (H:W) | Orientation | UCF-101? |
|-------------------|-----------------|-------------|----------|
| 4:3 (landscape) | **3:4** | LANDSCAPE (wider) | ✅ Use this! |
| 16:9 (widescreen) | **9:16** | LANDSCAPE (wider) | ✅ Or this! |
| 9:16 (portrait) | **16:9** | PORTRAIT (taller) | ❌ Wrong |

## The Fix

### For UCF-101 (640×480 landscape):

**Option 1: Use 3:4 aspect ratio** (closest to UCF-101's 640×480)
```python
aspect_ratio = "3:4"  # H:W = 3/4 = 0.75 → LANDSCAPE
# Results in: 554×738 (H×W) = 738 wide × 554 tall = LANDSCAPE ✓
```

**Option 2: Use 9:16 aspect ratio** (modern landscape)
```python
aspect_ratio = "9:16"  # H:W = 9/16 = 0.56 → LANDSCAPE  
# Wait, this is wrong! 9:16 in H:W notation is 9/16 = 0.56 < 1 → wider than tall → LANDSCAPE

Actually, let me recalculate...
```

Wait, I'm confusing myself again. Let me clarify:

## Correct Understanding

If H/W ratio:
- **H/W < 1** → Width > Height → **LANDSCAPE** (wider than tall)
- **H/W > 1** → Height > Width → **PORTRAIT** (taller than wide)
- **H/W = 1** → Square

So in Open-Sora:
- `"9:16"` → H/W = 9/16 = 0.56 < 1 → **LANDSCAPE** ✓
- `"3:4"` → H/W = 3/4 = 0.75 < 1 → **LANDSCAPE** ✓
- `"4:3"` → H/W = 4/3 = 1.33 > 1 → **PORTRAIT** ✗
- `"16:9"` → H/W = 16/9 = 1.78 > 1 → **PORTRAIT** ✗

## Recommended Fix for UCF-101

UCF-101 videos are **640×480** (W×H) = **480 tall × 640 wide** = **LANDSCAPE** with aspect W/H = 640/480 = 1.33.

In Open-Sora's H:W notation:
- UCF-101 H/W = 480/640 = 0.75
- Need: `aspect_ratio = "3:4"` (H:W = 3/4 = 0.75)

### Update All Configs:

```python
# OLD (WRONG - gives portrait):
aspect_ratio = "4:3"  # Comment said "640x480" but was actually portrait!

# NEW (CORRECT - gives landscape):
aspect_ratio = "3:4"  # 640x480 landscape, matches UCF-101
```

### Expected Output with 3:4:

From `ASPECT_RATIO_480P`:
```python
"0.75": (554, 738),  # H=554, W=738 → 738 wide × 554 tall = LANDSCAPE ✓
```

After VAE rounding (8x downscale then upscale):
- Latent: [15, 70, 93] (T, H, W)
- Decoded: 560 tall × 744 wide = LANDSCAPE ✓

Wait, that's still wrong. Let me recalculate the VAE math:
- Input: (554, 738) → (H, W)
- Latent: ((554-1)/8+1, (738-1)/8+1) = (70, 93) → (H, W)
- Decoded: (70×8, 93×8) = (560, 744) → (H, W)

So decoded tensor shape: `[C, T, H, W]` = `[3, 49, 560, 744]`

When saved with `save_sample`:
- `.permute(1, 2, 3, 0)` → `[T, H, W, C]` = `[49, 560, 744, 3]`
- `write_video_pyav`:
  - `stream.width = video.size(2)` = 744 ✓
  - `stream.height = video.size(1)` = 560 ✓
- Final video: **744 wide × 560 tall** = **LANDSCAPE** ✓

Perfect! So the fix is to use `aspect_ratio = "3:4"`.

## Summary

| Config | Aspect Ratio | H×W (Open-Sora) | Width×Height (Video) | Orientation | UCF-101 Match |
|--------|--------------|-----------------|----------------------|-------------|---------------|
| **Current** | `"4:3"` | 740×555 | 560×744 | ❌ PORTRAIT | ❌ No |
| **Fixed** | `"3:4"` | 554×738 | 744×560 | ✅ LANDSCAPE | ✅ Yes |

## Action Items

1. Update all configs: `aspect_ratio = "4:3"` → `aspect_ratio = "3:4"`
2. Re-run all tests with corrected aspect ratio
3. Verify output videos are landscape (wider than tall)
4. Compare with UCF-101 conditioning frames for aspect ratio match

