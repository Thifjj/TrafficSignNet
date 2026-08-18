#!/bin/bash
set -u

echo "TrafficSignNet ZCU104 environment check"
echo
command -v xdputil >/dev/null || { echo "ERROR: xdputil not found"; exit 1; }
python3 -c 'import cv2, numpy, vart, xir; print("Python VART dependencies: OK")' || exit 1
echo
xdputil query
echo
xdputil xmodel model/trafficsignnet_int8.xmodel -l
