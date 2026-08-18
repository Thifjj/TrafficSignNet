#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
./build/benchmark_zcu104 \
  --model ../model/trafficsignnet_int8.xmodel \
  --dataset ../dataset \
  --output-dir ../results_cpp \
  --warmup 1000 \
  --runs 5 \
  --seconds 60 \
  --workload-size 100 \
  --workload-runs 10 \
  --max-workers 4 \
  --sweep-runs 3 \
  --sweep-seconds 10 \
  --sweep-warmup 100 \
  --cold-runs 3
