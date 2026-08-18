#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build

CMAKE_ARGS=(
  -S .
  -B build
  -DCMAKE_BUILD_TYPE=Release
)

# The 2022.2 ZCU104 image can have protobuf installed while CMake's
# FindProtobuf fails to resolve Protobuf_LIBRARIES automatically.
if [[ -f /usr/lib/libprotobuf.so && -d /usr/include/google/protobuf ]]; then
  CMAKE_ARGS+=(
    -DProtobuf_LIBRARY=/usr/lib/libprotobuf.so
    -DProtobuf_INCLUDE_DIR=/usr/include
  )
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build build -j"$(nproc)"
echo
echo "Built: $(pwd)/build/benchmark_zcu104"
