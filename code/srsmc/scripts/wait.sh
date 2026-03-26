#!/bin/bash
# Wait until ALL GPUs have <1% memory usage for 60 consecutive minutes.
# This ensures we don't interfere with the GPU owner's workloads.

THRESHOLD_PCT=1
REQUIRED_MINUTES=60
CHECK_INTERVAL=60  # seconds

consecutive_idle=0

echo "=== GPU Idle Monitor ==="
echo "Waiting for all GPUs to be <${THRESHOLD_PCT}% memory usage for ${REQUIRED_MINUTES} consecutive minutes..."
echo "Checking every ${CHECK_INTERVAL}s. Started at $(date)"
echo ""

while true; do
    # Get memory usage percentage for each GPU
    all_idle=true
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.memory --format=csv,noheader,nounits)

    while IFS=, read -r idx mem_used mem_total mem_util; do
        idx=$(echo "$idx" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        pct=$((mem_used * 100 / mem_total))
        if [ "$pct" -ge "$THRESHOLD_PCT" ]; then
            all_idle=false
            break
        fi
    done <<< "$gpu_info"

    if $all_idle; then
        consecutive_idle=$((consecutive_idle + 1))
        remaining=$((REQUIRED_MINUTES - consecutive_idle))
        echo "[$(date '+%H:%M:%S')] All GPUs idle. ${consecutive_idle}/${REQUIRED_MINUTES} min (${remaining} min remaining)"
    else
        if [ "$consecutive_idle" -gt 0 ]; then
            echo "[$(date '+%H:%M:%S')] GPU activity detected. Resetting counter (was at ${consecutive_idle} min)."
        fi
        consecutive_idle=0
    fi

    if [ "$consecutive_idle" -ge "$REQUIRED_MINUTES" ]; then
        echo ""
        echo "=== All GPUs idle for ${REQUIRED_MINUTES} minutes. Proceeding! ==="
        echo "Time: $(date)"
        break
    fi

    sleep "$CHECK_INTERVAL"
done
