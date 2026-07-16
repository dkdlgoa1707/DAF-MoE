#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/run_hpo.sh <base-yaml> <phase2-search-yaml> [device]" >&2
    exit 2
fi

BASE_CONFIG=$1
SEARCH_SPACE=$2
DEVICE=${3:-cuda}

exec python runners/run_phase2.py hpo \
    --base-config "$BASE_CONFIG" \
    --search-space "$SEARCH_SPACE" \
    --device "$DEVICE"
