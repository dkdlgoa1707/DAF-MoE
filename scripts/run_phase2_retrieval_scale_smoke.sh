#!/usr/bin/env bash
set -euo pipefail

source /home/mars/anaconda3/etc/profile.d/conda.sh
conda activate daf_moe

python analysis/verify_phase2_retrieval_scale.py \
  --devices cuda:0 cuda:1 cuda:2 \
  "$@"
