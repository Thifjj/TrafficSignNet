#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
./build/benchmark_zcu104 \
  --model ../model/trafficsignnet_int8.xmodel \
  --dataset ../dataset \
  --output-dir ../results_cpp \
  --warmup 20 \
  --runs 1 \
  --seconds 2 \
  --workload-size 20 \
  --workload-runs 1 \
  --max-workers 2 \
  --sweep-runs 1 \
  --sweep-seconds 2 \
  --sweep-warmup 10 \
  --cold-runs 1 \
  --skip-max-confirmation
