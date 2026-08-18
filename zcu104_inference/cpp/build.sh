#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo
echo "Built: $(pwd)/build/benchmark_zcu104"
