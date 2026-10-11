#!/usr/bin/env bash
# Reproduces every number, table and figure in the paper. About 5-7 minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
OUT="${1:-results}"
python3 recon_bench.py --out "$OUT"
# Fingerprint of the deterministic result files only (a fixed list; environment.json and
# timing.json record the machine and wall-clock time, and SHA256SUMS never hashes itself).
( cd "$OUT" && sha256sum main_raw.csv main_summary.csv numbers.json recall_by_cause.csv routing.csv stress.csv > SHA256SUMS )
echo "results written to $OUT/"
