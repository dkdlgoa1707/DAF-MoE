#!/usr/bin/env bash
set -euo pipefail

exec python analysis/phase2_preflight.py "$@"
