# Benchmarks

All benchmark modes use batch 1 and synchronous single-thread calls. `model-only` measures an already prepared input through model execution; `host-to-host` includes host conversion/normalization and `argmax`. File I/O and image resize are outside the host-to-host timing.

## CPU FP32

```bash
python3 benchmarks/cpu/benchmark_tensorflow.py
```

This uses `models/keras/TrafficSignNet_FP32.h5` and writes to `results/cpu/tensorflow/`.

## CPU Vitis AI INT8

```bash
python3 benchmarks/cpu/benchmark_vitisai.py
```

This uses `models/keras/TrafficSignNet_INT8.h5` and writes to `results/cpu/vitisai/`. It requires the Vitis AI quantization runtime.

## NVIDIA GPU

```bash
python3 benchmarks/gpu/benchmark_nvidia.py --gpu-index 0
```

This uses the INT8 Keras model on a visible CUDA GPU and writes to `results/gpu/nvidia/`. It does not measure the ZCU104 DPU.

Results report FPS, mean/median latency, and tail percentiles. Compare runs only under similar load, frequency, temperature, runtime, warmup, and duration.
