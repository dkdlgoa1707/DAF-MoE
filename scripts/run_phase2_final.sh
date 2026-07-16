#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/run_phase2_final.sh <base-yaml> <search-yaml> [best-yaml] [device]" >&2
    exit 2
fi

BASE_CONFIG=$1
SEARCH_SPACE=$2
BEST_CONFIG=${3:-}
DEVICE=${4:-cuda}

ARGS=(
    final
    --base-config "$BASE_CONFIG"
    --search-space "$SEARCH_SPACE"
    --device "$DEVICE"
)
if [ -n "$BEST_CONFIG" ]; then
    ARGS+=(--best-config "$BEST_CONFIG")
fi

exec python runners/run_phase2.py "${ARGS[@]}"
