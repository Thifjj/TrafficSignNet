#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
./build/benchmark_zcu104 \
  --model ../model/trafficsignnet_int8.xmodel \
  --dataset ../dataset \
  --require-full-gtsrb-test \
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
  --full-e2e-runs 3 \
  --power-sample-ms 20 \
  --cold-runs 3 \
  --skip-cold
