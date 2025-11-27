#!/bin/bash
# Helper script to check T2V generation job status on cluster
# Run this on the cluster: bash check_t2v_status.sh

echo "=== Checking T2V Generation Job Status ==="
echo ""

# Check for recent SLURM output files
echo "Recent SLURM log files:"
ls -lt /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/ 2>/dev/null | head -10
echo ""

# Check for job in queue
echo "Jobs in queue (test_t2v*):"
squeue -u $USER | grep test_t2v
echo ""

# Check most recent error log
echo "=== Most Recent Error Log (last 50 lines) ==="
LATEST_ERR=$(ls -t /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_*.err 2>/dev/null | head -1)
if [ -n "$LATEST_ERR" ]; then
    echo "File: $LATEST_ERR"
    echo "---"
    tail -50 "$LATEST_ERR"
else
    echo "No error logs found"
fi
echo ""

# Check most recent output log
echo "=== Most Recent Output Log (last 50 lines) ==="
LATEST_OUT=$(ls -t /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_OUT" ]; then
    echo "File: $LATEST_OUT"
    echo "---"
    tail -50 "$LATEST_OUT"
else
    echo "No output logs found"
fi
echo ""

# Check if any videos were generated
echo "=== Generated Videos ==="
if [ -d "/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/results/debug/t2v_samples" ]; then
    ls -lh /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/results/debug/t2v_samples/
else
    echo "Output directory does not exist yet"
fi

