#!/bin/bash

# ========================================================
# Reproduction Script (15 Random Seeds)
# ========================================================
# Description:
#   Runs the training loop for 15 fixed random seeds (43-57)
#   to reproduce the statistical results reported in the paper.
#   Skips seeds that have already been computed.
#
# Usage: 
#   bash scripts/reproduce_results.sh <CONFIG_PATH> [GPU_ID]
#
# Example:
#   bash scripts/reproduce_results.sh configs/experiments/adult_daf_moe_best.yaml 0
# ========================================================

CONFIG_PATH=$1
GPU_ID=${2:-0}

# Validation
if [ -z "$CONFIG_PATH" ]; then
    echo "🚨 Usage: bash scripts/reproduce_results.sh <config_path> [gpu_id]"
    exit 1
fi

# Extract identifier from filename (e.g., adult_daf_moe_best.yaml -> adult_daf_moe)
FILENAME=$(basename -- "$CONFIG_PATH")
FILENAME_NO_EXT="${FILENAME%.*}"
BASE_IDENTIFIER=${FILENAME_NO_EXT%"_best"} 

# Seeds used in the paper (43 to 57)
SEEDS=$(seq 43 57)

echo "🔥 [Reproduction] Starting 15-Seed Run for '$BASE_IDENTIFIER' on GPU $GPU_ID"

for SEED in $SEEDS; do
    
    # Check if result already exists to allow resume capability
    # Expected output format: results/scores/{dataset}_{model}_seed{seed}.json
    RESULT_FILE="results/scores/${BASE_IDENTIFIER}_seed${SEED}.json"

    if [ -f "$RESULT_FILE" ]; then
        echo "⏩ [Skip] Seed $SEED already completed. ($RESULT_FILE)"
        continue
    fi

    echo "--------------------------------------------------------"
    echo "🌱 Running Seed: $SEED"
    echo "--------------------------------------------------------"
    
    # Run Training
    python train.py \
        --config "$CONFIG_PATH" \
        --gpu_ids "$GPU_ID" \
        --seed "$SEED" \
        --verbose 
        
    # Optional: Cooling down GPU
    # sleep 1
done

echo "✅ All 15 seeds finished for $CONFIG_PATH!"