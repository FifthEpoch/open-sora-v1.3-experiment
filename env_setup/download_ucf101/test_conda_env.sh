#!/bin/bash
# Diagnostic script to test conda environment activation in sbatch context
# This mimics what the sbatch script does to help debug activation issues

echo "========================================================================"
echo "Conda Environment Activation Test"
echo "========================================================================"
echo ""

# Set scratch environment variables (matching sbatch script)
SCRATCH_BASE="/scratch/wc3013"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"

echo "Step 1: Check if conda is available"
echo "-----------------------------------"
which conda
conda --version
echo ""

echo "Step 2: Load anaconda module"
echo "----------------------------"
module load anaconda3/2024.02
echo "✓ Module loaded"
echo ""

echo "Step 3: Source conda initialization"
echo "-----------------------------------"
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
echo "✓ Conda sourced"
echo ""

echo "Step 4: Deactivate any existing conda environment"
echo "-------------------------------------------------"
# If already in an env, deactivate first for clean activation
if [ -n "${CONDA_PREFIX}" ]; then
    echo "Currently in environment: ${CONDA_PREFIX}"
    echo "Deactivating for clean test..."
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true  # May need twice if nested
fi
echo "✓ Ready for fresh activation"
echo ""

echo "Step 5: Configure conda for custom paths"
echo "----------------------------------------"
# Use --prepend to ensure our path is searched first
conda config --prepend envs_dirs "${CONDA_ENVS_PATH}" 2>/dev/null || true
conda config --prepend pkgs_dirs "${CONDA_PKGS_DIRS}" 2>/dev/null || true
echo ""
echo "Current conda env_dirs:"
conda config --show envs_dirs
echo ""
echo "Current conda pkgs_dirs:"
conda config --show pkgs_dirs
echo ""

echo "Step 6: List available environments"
echo "-----------------------------------"
conda env list
echo ""

echo "Step 7: Check if opensora13 environment exists"
echo "----------------------------------------------"
if [ -d "${CONDA_ENVS_PATH}/opensora13" ]; then
    echo "✓ Environment found at: ${CONDA_ENVS_PATH}/opensora13"
    echo "  Directory contents:"
    ls -la "${CONDA_ENVS_PATH}/opensora13" | head -10
else
    echo "✗ Environment NOT found at: ${CONDA_ENVS_PATH}/opensora13"
    echo "  Please check if environment was created correctly"
    exit 1
fi
echo ""

echo "Step 8: Activate environment with full path"
echo "-------------------------------------------"
echo "Command: conda activate ${CONDA_ENVS_PATH}/opensora13"
conda activate "${CONDA_ENVS_PATH}/opensora13"

if [ $? -eq 0 ]; then
    echo "✓ Environment activated successfully"
else
    echo "✗ Failed to activate environment"
    exit 1
fi
echo ""

echo "Step 9: Verify activation"
echo "------------------------"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV}"
echo "Which python: $(which python)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"
echo ""

echo "Step 10: Test package imports"
echo "----------------------------"
python << 'EOF'
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("")

packages_to_test = [
    'torch',
    'av',
    'cv2',
    'numpy',
    'pandas',
    'tqdm',
    'datasets',
    'huggingface_hub'
]

print("Testing package imports:")
failed_imports = []
for pkg in packages_to_test:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError as e:
        print(f"  ✗ {pkg} - FAILED: {e}")
        failed_imports.append(pkg)

print("")
if failed_imports:
    print(f"✗ {len(failed_imports)} packages failed to import: {', '.join(failed_imports)}")
    sys.exit(1)
else:
    print("✓ All packages imported successfully")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ ALL TESTS PASSED"
    echo "========================================================================"
    echo "Your conda environment is correctly configured for sbatch jobs!"
else
    echo ""
    echo "========================================================================"
    echo "✗ SOME TESTS FAILED"
    echo "========================================================================"
    echo "Please review the errors above."
    exit 1
fi

