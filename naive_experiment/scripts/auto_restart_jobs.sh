#!/bin/bash
# ============================================================================
# Auto-restart script for jobs that get killed due to low GPU utilization
#
# The cluster kills jobs at EVEN hours (2pm, 4pm, 6pm, 8pm, 10pm, 12am, etc.)
# This script waits until just after each sweep, checks for killed jobs,
# and automatically restarts them from where they left off.
#
# Usage:
#   ./auto_restart_jobs.sh --account torch_pr_36_mren
#
# Run in tmux:
#   tmux new -s job_monitor
#   ./auto_restart_jobs.sh --account torch_pr_36_mren
#   # Detach with Ctrl+B, D
#   # Reattach with: tmux attach -t job_monitor
#
# This script will:
# 1. Submit initial jobs
# 2. Wait until 5 minutes after each even hour (when sweeps happen)
# 3. Check if jobs were killed
# 4. Resubmit with --start-from-video flag
# ============================================================================

set -u

# Configuration
MAX_RETRIES=50      # Maximum number of restarts per job
ACCOUNT=""
SWEEP_OFFSET_MINUTES=5  # Check this many minutes after the even hour

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ACCOUNT" ]; then
    echo "ERROR: --account is required"
    echo "Usage: $0 --account <slurm_account>"
    exit 1
fi

# Function to calculate seconds until next even hour + offset
get_seconds_until_next_check() {
    local current_hour=$(date +%H)
    local current_min=$(date +%M)
    local current_sec=$(date +%S)
    
    # Convert to total seconds since midnight
    local now_seconds=$((10#$current_hour * 3600 + 10#$current_min * 60 + 10#$current_sec))
    
    # Find next even hour
    local next_even_hour
    if [ $((10#$current_hour % 2)) -eq 0 ]; then
        # Current hour is even
        if [ $((10#$current_min)) -lt $SWEEP_OFFSET_MINUTES ]; then
            # Haven't passed the check time yet for this even hour
            next_even_hour=$((10#$current_hour))
        else
            # Already passed, wait for next even hour
            next_even_hour=$((10#$current_hour + 2))
        fi
    else
        # Current hour is odd, next even is +1
        next_even_hour=$((10#$current_hour + 1))
    fi
    
    # Handle day wrap
    if [ $next_even_hour -ge 24 ]; then
        next_even_hour=$((next_even_hour - 24))
        # Add a full day worth of seconds then subtract current time
        local target_seconds=$((next_even_hour * 3600 + SWEEP_OFFSET_MINUTES * 60))
        echo $((86400 - now_seconds + target_seconds))
    else
        local target_seconds=$((next_even_hour * 3600 + SWEEP_OFFSET_MINUTES * 60))
        echo $((target_seconds - now_seconds))
    fi
}

# Function to format next check time
get_next_check_time() {
    local seconds_until=$1
    date -d "+${seconds_until} seconds" "+%H:%M:%S" 2>/dev/null || \
    date -v+${seconds_until}S "+%H:%M:%S" 2>/dev/null || \
    echo "in ${seconds_until}s"
}

# Job configurations: (name, sbatch_file, output_dir)
declare -A JOBS
JOBS["100steps_5e5"]="naive_experiment/scripts/run_100steps_5e5.sbatch|naive_experiment/results/results_100steps_5e5"
JOBS["50steps_5e5"]="naive_experiment/scripts/run_50steps_5e5.sbatch|naive_experiment/results/results_50steps_5e5"
JOBS["50steps_1e5"]="naive_experiment/scripts/run_50steps_1e5.sbatch|naive_experiment/results/results_50steps_1e5"

# Track job IDs and retry counts
declare -A JOB_IDS
declare -A RETRY_COUNTS
declare -A COMPLETED

# Initialize
for job_name in "${!JOBS[@]}"; do
    JOB_IDS[$job_name]=""
    RETRY_COUNTS[$job_name]=0
    COMPLETED[$job_name]=false
done

# Function to get last processed video index from progress.json or logs
get_last_video_idx() {
    local output_dir="$1"
    local job_id="$2"
    
    # First, try to read from progress.json (most reliable)
    # progress.json is a list like: [{"video_idx": 0, "status": "completed"}, ...]
    local progress_file="${output_dir}/progress.json"
    if [ -f "$progress_file" ]; then
        local last_idx=$(python3 -c "
import json
try:
    with open('$progress_file') as f:
        progress = json.load(f)
    # progress is a list of dicts with 'video_idx' and 'status'
    completed = [p['video_idx'] for p in progress if p.get('status') == 'completed']
    print(max(completed) + 1 if completed else 0)
except Exception as e:
    print(0)
" 2>/dev/null)
        if [ -n "$last_idx" ] && [ "$last_idx" != "0" ]; then
            echo "$last_idx"
            return
        fi
    fi
    
    # Fallback: parse stderr log
    local err_log="naive_experiment/scripts/slurm_log/*${job_id}.err"
    local latest_err=$(ls -t $err_log 2>/dev/null | head -1)
    if [ -n "$latest_err" ] && [ -f "$latest_err" ]; then
        # Look for "Processing video X:" or "video X completed" pattern
        local last_idx=$(grep -oP 'Processing video \K\d+|video \K\d+(?= marked as completed)' "$latest_err" | tail -1)
        if [ -n "$last_idx" ]; then
            echo $((last_idx + 1))  # Start from next video
            return
        fi
    fi
    
    echo "0"
}

# Function to check if job is still running
is_job_running() {
    local job_id="$1"
    if [ -z "$job_id" ]; then
        return 1
    fi
    squeue -j "$job_id" -h 2>/dev/null | grep -q "$job_id"
}

# Function to check if job completed successfully
is_job_completed() {
    local output_dir="$1"
    # Check if all 100 videos are done
    local progress_file="${output_dir}/progress.json"
    if [ -f "$progress_file" ]; then
        local completed_count=$(python3 -c "
import json
try:
    with open('$progress_file') as f:
        progress = json.load(f)
    # progress is a list of dicts
    completed = [p for p in progress if p.get('status') == 'completed']
    print(len(completed))
except:
    print(0)
" 2>/dev/null)
        if [ "$completed_count" == "100" ]; then
            return 0
        fi
    fi
    return 1
}

# Function to submit a job
submit_job() {
    local job_name="$1"
    local start_from="$2"
    
    IFS='|' read -r sbatch_file output_dir <<< "${JOBS[$job_name]}"
    
    # Create a temporary modified sbatch with --start-from-video
    local temp_sbatch=$(mktemp)
    
    # Copy original sbatch and modify the python command
    if [ "$start_from" -gt 0 ]; then
        # Add --start-from-video flag
        sed "s/--skip-baseline/--skip-baseline --start-from-video ${start_from}/" "$sbatch_file" > "$temp_sbatch"
    else
        cp "$sbatch_file" "$temp_sbatch"
    fi
    
    # Submit job
    local job_id=$(sbatch --account="$ACCOUNT" "$temp_sbatch" 2>&1 | grep -oP '\d+$')
    rm "$temp_sbatch"
    
    if [ -n "$job_id" ]; then
        echo "$job_id"
        return 0
    else
        return 1
    fi
}

# Main loop
echo "========================================"
echo "Auto-Restart Job Monitor"
echo "========================================"
echo "Account: $ACCOUNT"
echo "Sweep times: Every even hour (2pm, 4pm, 6pm, ...)"
echo "Check time: ${SWEEP_OFFSET_MINUTES} minutes after each sweep"
echo "Jobs to monitor: ${!JOBS[*]}"
echo "========================================"
echo ""

# Initial submission
echo "[$(date)] Submitting initial jobs..."
for job_name in "${!JOBS[@]}"; do
    IFS='|' read -r sbatch_file output_dir <<< "${JOBS[$job_name]}"
    
    # Check if we should resume from existing progress
    start_from=0
    if [ -d "$output_dir" ]; then
        start_from=$(get_last_video_idx "$output_dir" "")
        if [ "$start_from" -gt 0 ]; then
            echo "  Found existing progress for $job_name, resuming from video $start_from"
        fi
    fi
    
    job_id=$(submit_job "$job_name" "$start_from")
    if [ -n "$job_id" ]; then
        JOB_IDS[$job_name]="$job_id"
        echo "  Submitted $job_name: Job ID $job_id (start_from=$start_from)"
    else
        echo "  ERROR: Failed to submit $job_name"
    fi
done

echo ""
echo "[$(date)] Monitoring started. Press Ctrl+C to stop."
echo ""

# Do an immediate check first (in case jobs were already killed)
check_and_restart_jobs() {
    for job_name in "${!JOBS[@]}"; do
        # Skip if already marked complete
        if [ "${COMPLETED[$job_name]}" == "true" ]; then
            continue
        fi
        
        IFS='|' read -r sbatch_file output_dir <<< "${JOBS[$job_name]}"
        job_id="${JOB_IDS[$job_name]}"
        
        # Check if job completed successfully
        if is_job_completed "$output_dir"; then
            echo "[$(date)] ✓ $job_name COMPLETED (all 100 videos done)"
            COMPLETED[$job_name]=true
            continue
        fi
        
        # Check if job is still running
        if is_job_running "$job_id"; then
            echo "[$(date)] $job_name (ID: $job_id) is still running"
            continue
        fi
        
        # Job is not running and not complete - needs restart
        echo "[$(date)] Job $job_name (ID: $job_id) was killed. Restarting..."
        
        # Get last processed video
        last_idx=$(get_last_video_idx "$output_dir" "$job_id")
        echo "  Last completed video: $((last_idx - 1)), will restart from: $last_idx"
        
        # Check retry count
        retry_count="${RETRY_COUNTS[$job_name]}"
        if [ "$retry_count" -ge "$MAX_RETRIES" ]; then
            echo "  ERROR: Max retries ($MAX_RETRIES) reached for $job_name"
            COMPLETED[$job_name]=true
            continue
        fi
        
        # Resubmit
        echo "  Resubmitting $job_name (retry $((retry_count + 1))/$MAX_RETRIES)..."
        new_job_id=$(submit_job "$job_name" "$last_idx")
        
        if [ -n "$new_job_id" ]; then
            JOB_IDS[$job_name]="$new_job_id"
            RETRY_COUNTS[$job_name]=$((retry_count + 1))
            echo "  ✓ Resubmitted as Job ID: $new_job_id"
        else
            echo "  ERROR: Failed to resubmit $job_name"
        fi
    done
}

# Monitor loop - wait for sweep times
while true; do
    # Check if all jobs are done
    all_done=true
    for job_name in "${!JOBS[@]}"; do
        if [ "${COMPLETED[$job_name]}" != "true" ]; then
            all_done=false
            break
        fi
    done
    
    if [ "$all_done" == "true" ]; then
        echo ""
        echo "========================================"
        echo "[$(date)] All jobs completed!"
        echo "========================================"
        break
    fi
    
    # Calculate time until next check (5 min after next even hour)
    seconds_until=$(get_seconds_until_next_check)
    next_time=$(get_next_check_time "$seconds_until")
    
    echo ""
    echo "[$(date)] Waiting for next sweep at even hour..."
    echo "  Next check in: ${seconds_until} seconds (at ~${next_time})"
    echo "  Jobs running:"
    for job_name in "${!JOBS[@]}"; do
        if [ "${COMPLETED[$job_name]}" != "true" ]; then
            job_id="${JOB_IDS[$job_name]}"
            if is_job_running "$job_id"; then
                echo "    - $job_name (ID: $job_id): RUNNING"
            else
                echo "    - $job_name (ID: $job_id): NOT RUNNING (will restart)"
            fi
        fi
    done
    
    # Sleep until next check time
    sleep "$seconds_until"
    
    echo ""
    echo "========================================"
    echo "[$(date)] Sweep time - checking job status..."
    echo "========================================"
    
    # Give jobs a moment to fully terminate after sweep
    sleep 30
    
    # Check and restart killed jobs
    check_and_restart_jobs
done

echo ""
echo "Summary:"
for job_name in "${!JOBS[@]}"; do
    echo "  $job_name: ${RETRY_COUNTS[$job_name]} restarts"
done

