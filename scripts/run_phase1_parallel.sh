#!/usr/bin/env bash
# DAF-MoE v1.5 Phase 1: Option B parallel launcher.
# Per GPU: seven variants concurrently. Across seeds: sequential waves.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs/phase1_v15

DATASETS_GPU=("california:0" "adult:1" "mimic4:2")
VARIANTS=(M0 M1 M2 M3 M4 M5 M6)
SEEDS=(42 43 44 45 46)

START_TS=$(date +%s)
TOTAL_FAILURES=0

for seed in "${SEEDS[@]}"; do
    echo "==================================================="
    echo "[Wave] Seed $seed start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==================================================="
    PIDS=()
    LABELS=()

    for dg in "${DATASETS_GPU[@]}"; do
        dataset="${dg%:*}"
        gpu="${dg#*:}"
        for variant in "${VARIANTS[@]}"; do
            logfile="logs/phase1_v15/${dataset}_${variant}_seed${seed}.log"
            python runners/run_phase1_v15.py \
                --datasets "$dataset" \
                --variants "$variant" \
                --seeds "$seed" \
                --gpu-id "$gpu" \
                > "$logfile" 2>&1 &
            PIDS+=("$!")
            LABELS+=("${dataset}/${variant}/seed${seed}")
        done
    done

    echo "[Wave] Launched ${#PIDS[@]} subprocesses for seed $seed"
    WAVE_FAILURES=0
    for index in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$index]}"; then
            echo "[Wave][Fail] ${LABELS[$index]} (pid=${PIDS[$index]})"
            WAVE_FAILURES=$((WAVE_FAILURES + 1))
        fi
    done
    TOTAL_FAILURES=$((TOTAL_FAILURES + WAVE_FAILURES))
    echo "[Wave] Seed $seed complete: $(date '+%Y-%m-%d %H:%M:%S') failures=$WAVE_FAILURES"
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo ""
echo "=================================================="
echo "[Done] Phase 1 complete. wall_clock=${ELAPSED}s launcher_failures=${TOTAL_FAILURES}"
echo "  Logs:     logs/phase1_v15/"
echo "  Results:  results/phase1_v15/"
echo "  Failures: results/phase1_v15/failures__*.log"
echo "=================================================="

if (( TOTAL_FAILURES > 0 )); then
    exit 1
fi
