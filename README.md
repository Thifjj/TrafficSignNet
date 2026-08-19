# TrafficSignNet

TrafficSignNet is a complete GTSRB traffic-sign classification project. It contains the dataset preparation flow, model artifacts, TensorFlow/Vitis AI CPU benchmarks, NVIDIA GPU benchmarks, and a Vitis AI deployment for the ZCU104.

## Repository map

- `data/gtsrb/`: downloaded archives, extracted raw files, and processed train/validation/test splits.
- `training/`: dataset preparation and TensorFlow-compatible model reconstruction.
- `optimization/`: Vitis AI INT8 quantization.
- `models/`: Keras and compiled accelerator artifacts.
- `benchmarks/cpu/`: FP32 TensorFlow and Vitis AI INT8 CPU benchmarks.
- `benchmarks/gpu/`: NVIDIA CUDA benchmark.
- `deploy/zcu104/`: board-side VART inference and C++ benchmark.
- `results/`: benchmark measurements grouped by platform.
- `docs/`: focused project documentation.

Start with [docs/README.md](docs/README.md).
