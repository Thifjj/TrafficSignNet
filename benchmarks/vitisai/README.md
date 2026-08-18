# TrafficSignNet Vitis AI CPU benchmark

This is the Vitis AI quantized counterpart to
`benchmarks/tensorflow/benchmark_cpu_model_and_host.py`. It loads
`models/keras/TrafficSignNet_INT8.h5` using the Vitis quantization scope and measures:

- **model-only**: normalized `float32` input through the quantized model;
- **host-to-host**: host `uint8` input, conversion and normalization, quantized
  model inference, and `argmax`.

Run it from the project root using the Python 3.8.6 / TensorFlow 2.12 Vitis AI
environment:

```bash
python3 benchmarks/vitisai/benchmark_cpu_model_and_host.py
```

The default workload matches the original benchmark: 1,000 warmups, five runs,
and 60 seconds per run. For a quick smoke test:

```bash
python3 benchmarks/vitisai/benchmark_cpu_model_and_host.py \
  --warmup 2 --runs 1 --seconds 1
```

Results are written to `results/vitisai/cpu_model_only.json` and
`results/vitisai/cpu_host_to_host.json`. Use `--model` and `--results-dir` to
override those paths.

## NVIDIA GeForce RTX 4070 SUPER

`benchmark_nvidia_gpu.py` uses the same two benchmark modes through CUDA. It
requires a real TensorFlow GPU device and refuses to fall back silently to CPU:

```bash
python3 benchmarks/vitisai/benchmark_nvidia_gpu.py \
  --warmup 2 --runs 1 --seconds 1
```

The script selects a device whose TensorFlow name contains `4070`. If more than
one NVIDIA GPU is present and automatic selection is ambiguous, it prints all
devices so the correct index can be supplied explicitly:

```bash
python3 benchmarks/vitisai/benchmark_nvidia_gpu.py --gpu-index 0
```

Use `--log-device-placement` to inspect TensorFlow operation placement. Output
is written below `results/vitisai_nvidia_gpu/`.

The RTX 4070 SUPER is a dedicated NVIDIA GPU, not an integrated GPU. With the
requested TensorFlow 2.12 environment, use a compatible NVIDIA driver, CUDA
11.8, and cuDNN 8.6. The GPU must appear in
`tf.config.list_physical_devices("GPU")`; a CPU-only TensorFlow package cannot
run this benchmark.
