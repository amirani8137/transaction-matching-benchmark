#!/usr/bin/env bash
# Reproduces every number, table and figure in the paper. ~5-10 minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
python3 recon_bench.py --out "${1:-results}"
# fingerprint of the result files (environment.json excluded: it records the machine and runtime)
( cd "${1:-results}" && sha256sum $(ls | grep -v environment.json) > SHA256SUMS )
echo "results written to ${1:-results}/"
