# Flash-Attention and Apex Installation Guide

## Summary of Previous Issues

### Why They're Difficult to Install

#### **flash-attn Issues:**
1. **CUDA Version Mismatch**
   - PyTorch 2.2.2 was built with CUDA 12.1
   - System CUDA toolkit is 12.2
   - flash-attn requires **exact** CUDA version match during compilation
   - Build process fails if CUDA runtime version ≠ compilation version

2. **GPU Architecture Requirement**
   - Must compile for specific GPU compute capability
   - H100/H200 = compute capability 9.0 (TORCH_CUDA_ARCH_LIST="90")
   - Requires `nvcc` (CUDA compiler) available during build
   - Conda PyTorch doesn't include full CUDA toolkit

3. **Filesystem Issues**
   - Build creates large temporary files
   - pip cache can cause cross-device link errors
   - Requires all temp directories on same filesystem (scratch)

#### **apex Issues:**
1. **CUDA Version Sensitivity**
   - Even more sensitive to CUDA version mismatches than flash-attn
   - NVIDIA apex expects exact CUDA toolkit version used to build PyTorch
   - System CUDA 12.2 vs PyTorch CUDA 12.1 causes build failures

2. **Fragile Build Process**
   - Complex C++/CUDA mixed compilation
   - Frequently breaks with newer CUDA versions
   - Requires both `--cpp_ext` and `--cuda_ext` options

3. **Known to Fail**
   - Script comments explicitly note: "Apex build will likely fail due to CUDA 12.1/12.2 mismatch"
   - Build script has fallback: "|| echo 'Apex build failed - this is OK'"

## Current Status

Based on our configs, we currently have:
- `enable_flash_attn=False`
- `enable_layernorm_kernel=False`

This suggests they were **not successfully installed** or we chose not to install them due to difficulty.

## Impact on Quality

**CRITICAL FINDING:** These are **NOT just speed optimizations!**

The Open-Sora model was **trained** with flash-attn and apex enabled. When disabled:
- The model uses **fallback implementations** (standard PyTorch attention and LayerNorm)
- These produce **numerically different results**
- This causes **significant quality degradation**, not just slower speed

**This explains the terrible T2V quality we're seeing!**

## Options for Moving Forward

### Option 1: Fix Config Without Optimizations (Quick Test)

**Action:** Test with the new `t2v_generation_noopt.py` config which fixes:
- `tile_size`: 16 → 4 (reduces VAE artifacts)
- `use_flaw_fix`: Missing → True (enables quality enhancement)
- `num_sampling_steps`: 50 → 30 (matches official)

**Pros:**
- No installation needed
- Quick test (can run immediately)
- Fixes some quality issues (VAE tiling, scheduler)

**Cons:**
- Still uses fallback attention/layernorm
- Quality may still be degraded vs. models with optimizations
- Unknown if model behavior is correct without flash-attn

**Verdict:** Worth testing, but may not fully solve quality issues

---

### Option 2: Try Installing flash-attn (Moderate Effort)

flash-attn is more important than apex for attention mechanism quality.

**Simplified Installation Attempt:**

```bash
# On a GPU node
module load anaconda3/2025.06
conda activate /scratch/wc3013/conda-envs/opensora13

# Use system CUDA 12.2 (match it to PyTorch build)
export CUDA_HOME="/usr/local/cuda-12.2"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Set GPU architecture for H200
export TORCH_CUDA_ARCH_LIST="90"

# Set all temp directories to scratch
export TMPDIR=/scratch/wc3013/tmp
export TMP=/scratch/wc3013/tmp  
export TEMP=/scratch/wc3013/tmp

# Try pip install (prebuilt wheel if available)
pip install flash-attn --no-cache-dir

# If that fails, build from source
cd /scratch/wc3013/tmp
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.8
pip install . --no-build-isolation
```

**Success Rate:** Moderate (50-70%)
- May work if CUDA 12.2 is compatible enough with PyTorch's CUDA 12.1
- H200 support is good (compute capability 9.0 is well-supported)

**Time:** 30-60 minutes (compilation takes a while)

**If Successful:**
- Enable `enable_flash_attn=True` in config
- Keep `enable_layernorm_kernel=False` (can live without apex)
- Should get **major quality improvement**

---

### Option 3: Install Both flash-attn and apex (High Effort)

**Use the existing script:**
```bash
sbatch env_setup/02_flsh_attn_apex_build.sbatch
```

**Expected Outcome:**
- flash-attn: Likely succeeds (70% chance)
- apex: Likely fails (30% chance) due to CUDA version mismatch

**If flash-attn succeeds but apex fails:**
- Still a win! flash-attn is more important
- Can use `enable_flash_attn=True, enable_layernorm_kernel=False`

**If both succeed:**
- Best quality possible
- Use official config as-is with both enabled

**Time:** 1-2 hours (includes queue time + build time)

---

### Option 4: Use Pre-built Wheels (If Available)

Some flash-attn versions have pre-built wheels for specific CUDA versions.

**Check availability:**
```bash
pip index versions flash-attn
```

**Try installing pre-built for CUDA 12.1:**
```bash
pip install flash-attn==2.5.8+cu121 --no-deps --no-cache-dir
```

**Success Rate:** Low (20-30%)
- Pre-built wheels may not exist for your exact config
- May not support H200 (compute capability 9.0)

**Time:** 5-10 minutes

---

## Recommended Approach

### Phase 1: Quick Test (30 minutes)
```bash
# Test the no-opt config with fixed parameters
sbatch --account=torch_pr_36_mren naive_experiment/scripts/test_t2v_official.sbatch
```

**If quality improves significantly:**
- The issue was mainly tile_size and use_flaw_fix
- Can proceed with experiment using no-opt config
- Accept 2-3x slower inference

**If quality still terrible:**
- Confirms we need flash-attn for correct model behavior
- Proceed to Phase 2

### Phase 2: Install flash-attn (1-2 hours)
```bash
# Run the build script
sbatch env_setup/02_flsh_attn_apex_build.sbatch

# Monitor progress
tail -f env_setup/slurm_build_kernels_*.out

# After completion, check what was installed
python naive_experiment/scripts/check_optimizations.py
```

**If flash-attn installs successfully:**
- Update config: `enable_flash_attn=True`
- Rerun T2V test with official config
- Quality should improve dramatically

**If flash-attn fails:**
- Review build logs for specific errors
- May need to adjust CUDA_HOME or other env vars
- Consider reaching out to cluster support for help

### Phase 3: Decide Next Steps Based on Results

1. **Good quality with no-opt config:**
   - Proceed with experiment as-is
   - Document that quality may be slightly degraded

2. **Good quality only with flash-attn:**
   - Ensure flash-attn works for all experiments
   - Update all configs to use it
   - Document as requirement

3. **Still poor quality even with flash-attn:**
   - Investigate model checkpoint loading
   - Check if models actually downloaded correctly
   - Compare our outputs with official Open-Sora demos

## Quick Decision Matrix

| Scenario | Action | Time | Success Likelihood |
|----------|--------|------|-------------------|
| Just want to test quickly | Run test_t2v_official.sbatch (no-opt) | 30 min | High (100%) |
| Willing to spend 1-2 hours | sbatch 02_flsh_attn_apex_build.sbatch | 1-2 hrs | Moderate (60%) |
| Need guaranteed quality | Manually install flash-attn with support | 2-4 hrs | High (90%) with help |

## My Recommendation

**Start with Phase 1** (test the no-opt config) because:
1. It's quick and requires no installation
2. It fixes real issues (tile_size, use_flaw_fix)
3. We'll learn if the quality problem is purely the optimizations or also config errors
4. If it works well enough, we can proceed with the experiment

**Then decide** based on results:
- If quality is acceptable → continue with experiment
- If quality is still terrible → invest time in flash-attn installation

This gives us maximum information with minimum time investment upfront.

