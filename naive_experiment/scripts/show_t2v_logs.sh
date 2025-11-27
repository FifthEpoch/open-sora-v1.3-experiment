#!/bin/bash
# Quick script to show the latest T2V logs
# Usage: bash show_t2v_logs.sh [err|out|both]

MODE="${1:-both}"

DEBUG_DIR="/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug"

if [ "$MODE" = "err" ] || [ "$MODE" = "both" ]; then
    LATEST_ERR=$(ls -t $DEBUG_DIR/test_t2v_*.err 2>/dev/null | head -1)
    if [ -n "$LATEST_ERR" ]; then
        echo "=========================================="
        echo "ERROR LOG: $(basename $LATEST_ERR)"
        echo "=========================================="
        cat "$LATEST_ERR"
        echo ""
    else
        echo "No error logs found in $DEBUG_DIR"
    fi
fi

if [ "$MODE" = "out" ] || [ "$MODE" = "both" ]; then
    LATEST_OUT=$(ls -t $DEBUG_DIR/test_t2v_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_OUT" ]; then
        echo "=========================================="
        echo "OUTPUT LOG: $(basename $LATEST_OUT)"
        echo "=========================================="
        cat "$LATEST_OUT"
        echo ""
    else
        echo "No output logs found in $DEBUG_DIR"
    fi
fi

