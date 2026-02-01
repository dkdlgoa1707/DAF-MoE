#!/bin/bash

# ========================================================
# HPO Execution Script using Optuna
# ========================================================
# Usage: 
#   bash scripts/run_hpo.sh <BASE_CONFIG> <HPO_CONFIG> [METRIC] [TRIALS] [GPU_ID]
#
# Example:
#   bash scripts/run_hpo.sh configs/experiments/adult_daf_moe.yaml configs/hpo/daf_moe.yaml acc 50 0
# ========================================================

# Parse Arguments
BASE_CONFIG=$1
HPO_CONFIG=$2
METRIC=${3:-"acc"}
TRIALS=${4:-30}
GPU_IDS=${5:-"0"}

# Shift arguments to capture extra args (if any)
shift 5
EXTRA_ARGS=$@

# Validation
if [ -z "$BASE_CONFIG" ] || [ -z "$HPO_CONFIG" ]; then
    echo "❌ [Error] Missing configuration files."
    echo "👉 Usage: bash scripts/run_hpo.sh <base_yaml> <hpo_yaml> [metric] [trials] [gpu] [extra_args]"
    exit 1
fi

echo "================================================================"
echo "⚡ Hyperparameter Tuning Start"
echo "📄 Base Config : $BASE_CONFIG"
echo "🎛️  HPO Config  : $HPO_CONFIG"
echo "🎯 Metric      : $METRIC"
echo "🔄 Trials      : $TRIALS"
echo "⚙️  GPU ID      : $GPU_IDS"
echo "================================================================"

# Execute tune.py
PYTHONPATH=. python tune.py \
    --base_config "$BASE_CONFIG" \
    --hpo_config "$HPO_CONFIG" \
    --metric "$METRIC" \
    --trials "$TRIALS" \
    --gpu_ids "$GPU_IDS" \
    $EXTRA_ARGS