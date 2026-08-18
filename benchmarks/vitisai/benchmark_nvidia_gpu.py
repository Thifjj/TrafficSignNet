#!/usr/bin/env python3
"""Benchmark Vitis AI TrafficSignNet on an NVIDIA CUDA GPU.

Python 3.8 and TensorFlow 2.12 compatible. Unlike the CPU benchmark, this
script never silently falls back when TensorFlow cannot expose a GPU.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from tensorflow_model_optimization.quantization.keras import vitis_quantize
except ImportError as exc:
    raise ImportError(
        "O pacote Vitis AI quantizer e necessario para carregar o modelo INT8."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_DIR / "models" / "keras" / "TrafficSignNet_INT8.h5"
DEFAULT_RESULTS = PROJECT_DIR / "results" / "vitisai_nvidia_gpu"
INPUT_SHAPE = (1, 32, 32, 3)
NVIDIA_MARKERS = ("nvidia", "geforce", "rtx", "4070")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark do TrafficSignNet quantizado em GPU NVIDIA/CUDA."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Indice em tf.config.list_physical_devices('GPU'). Por padrao, "
        "seleciona a RTX 4070 SUPER ou a unica GPU NVIDIA disponivel.",
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--log-device-placement", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1 or args.seconds <= 0:
        parser.error("use warmup >= 0, runs >= 1 e seconds > 0")
    return args


def gpu_description(device):
    details = tf.config.experimental.get_device_details(device)
    return str(details.get("device_name", device.name))


def select_nvidia_gpu(requested_index):
    if not tf.test.is_built_with_cuda():
        raise RuntimeError(
            "Este TensorFlow nao foi compilado com CUDA. Instale a distribuicao "
            "GPU do TensorFlow 2.12 antes de executar o benchmark."
        )

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError(
            "TensorFlow nao detectou GPU. Este teste nao permite fallback para CPU. "
            "Instale o TensorFlow 2.12 com CUDA 11.8 e cuDNN 8.6 compativeis."
        )

    print("GPUs fisicas detectadas pelo TensorFlow:")
    for index, device in enumerate(devices):
        print("  [{}] {} ({})".format(index, gpu_description(device), device.name))

    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(devices):
            raise ValueError("--gpu-index fora do intervalo detectado")
        selected_index = requested_index
    else:
        rtx_4070_matches = []
        nvidia_matches = []
        for index, device in enumerate(devices):
            name = gpu_description(device).lower()
            if "4070" in name:
                rtx_4070_matches.append(index)
            if any(marker in name for marker in NVIDIA_MARKERS):
                nvidia_matches.append(index)
        if len(rtx_4070_matches) == 1:
            selected_index = rtx_4070_matches[0]
        elif len(nvidia_matches) == 1:
            selected_index = nvidia_matches[0]
        elif len(devices) == 1:
            selected_index = 0
        else:
            raise RuntimeError(
                "Nao foi possivel identificar a RTX 4070 SUPER sem ambiguidade. "
                "Execute novamente com --gpu-index N usando a lista acima."
            )

    selected = devices[selected_index]
    selected_name = gpu_description(selected)
    if not any(marker in selected_name.lower() for marker in NVIDIA_MARKERS):
        raise RuntimeError(
            "O dispositivo selecionado ({}) nao parece ser uma GPU NVIDIA."
            .format(selected_name)
        )
    # Expose only the selected physical GPU. It becomes logical /GPU:0.
    tf.config.set_visible_devices(selected, "GPU")
    tf.config.experimental.set_memory_growth(selected, True)
    # Unsupported operations must fail instead of being measured on the CPU.
    tf.config.set_soft_device_placement(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError("Falha ao criar o dispositivo logico da GPU selecionada")
    return selected_index, selected_name, logical[0].name


def load_model(model_path):
    if not model_path.is_file():
        raise FileNotFoundError("Modelo nao encontrado: {}".format(model_path))
    with tf.device("/GPU:0"):
        with vitis_quantize.quantize_scope():
            return tf.keras.models.load_model(str(model_path), compile=False)


def make_inference_functions(model):
    @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)])
    def model_only(value):
        with tf.device("/GPU:0"):
            return model(value, training=False)

    @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.uint8)])
    def host_to_host(value):
        with tf.device("/GPU:0"):
            normalized = tf.cast(value, tf.float32) / 255.0
            logits = model(normalized, training=False)
            return tf.argmax(logits, axis=1, output_type=tf.int32)

    return model_only, host_to_host


def benchmark(name, infer, value, warmup, run_count, seconds):
    print("\n{}: aquecimento de {} inferencias".format(name, warmup))
    for _ in range(warmup):
        infer(value).numpy()

    runs = []
    for run_id in range(1, run_count + 1):
        latencies = []
        run_start = time.perf_counter_ns()
        deadline = run_start + int(seconds * 1e9)
        while time.perf_counter_ns() < deadline:
            start = time.perf_counter_ns()
            infer(value).numpy()
            latencies.append(time.perf_counter_ns() - start)

        elapsed = (time.perf_counter_ns() - run_start) / 1e9
        values = np.asarray(latencies, dtype=np.float64) / 1e6
        result = {
            "run": run_id,
            "inferences": len(values),
            "elapsed_seconds": elapsed,
            "fps": float(len(values) / elapsed),
            "latency_ms": {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "p99_9": float(np.percentile(values, 99.9)),
            },
        }
        runs.append(result)
        print(
            "Run {}: {:.2f} FPS | media {:.4f} ms | P99 {:.4f} ms".format(
                run_id,
                result["fps"],
                result["latency_ms"]["mean"],
                result["latency_ms"]["p99"],
            )
        )

    fps = np.asarray([item["fps"] for item in runs])

    def average(metric):
        return float(np.mean([item["latency_ms"][metric] for item in runs]))

    return {
        "mode": name,
        "batch_size": 1,
        "warmup": warmup,
        "runs": run_count,
        "seconds_per_run": seconds,
        "fps_mean": float(fps.mean()),
        "fps_median": float(np.median(fps)),
        "fps_min": float(fps.min()),
        "fps_max": float(fps.max()),
        "latency_mean_ms": average("mean"),
        "latency_median_ms": average("median"),
        "p90_ms": average("p90"),
        "p95_ms": average("p95"),
        "p99_ms": average("p99"),
        "p99_9_ms": average("p99_9"),
        "runs_data": runs,
    }


def main():
    args = parse_args()
    if args.log_device_placement:
        tf.debugging.set_log_device_placement(True)

    gpu_index, gpu_name, logical_name = select_nvidia_gpu(args.gpu_index)
    model = load_model(args.model.resolve())
    model_only, host_to_host = make_inference_functions(model)

    float_input = tf.random.stateless_uniform(
        INPUT_SHAPE, seed=(42, 42), dtype=tf.float32
    )
    uint8_input = np.random.default_rng(42).integers(
        0, 256, size=INPUT_SHAPE, dtype=np.uint8
    )

    # Trace once, then prove that TensorFlow created a GPU-backed result. This
    # prevents a CPU-only TensorFlow installation from producing misleading data.
    probe = model_only(float_input)
    if "GPU" not in probe.device.upper():
        raise RuntimeError(
            "A saida foi colocada em {}, nao na GPU. Use "
            "--log-device-placement para diagnosticar.".format(probe.device)
        )

    common = {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "tensorflow_cuda_build": tf.test.is_built_with_cuda(),
        "cuda_version": str(tf.sysconfig.get_build_info().get("cuda_version", "")),
        "cudnn_version": str(
            tf.sysconfig.get_build_info().get("cudnn_version", "")
        ),
        "physical_gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "logical_device": logical_name,
        "model": str(args.model.resolve()),
    }
    model_result = benchmark(
        "VITIS-AI GPU MODEL-ONLY",
        model_only,
        float_input,
        args.warmup,
        args.runs,
        args.seconds,
    )
    host_result = benchmark(
        "VITIS-AI GPU HOST-TO-HOST UINT8",
        host_to_host,
        uint8_input,
        args.warmup,
        args.runs,
        args.seconds,
    )
    model_result.update(common)
    host_result.update(common)

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    model_file = results_dir / "gpu_model_only.json"
    host_file = results_dir / "gpu_host_to_host.json"
    with model_file.open("w") as output:
        json.dump(model_result, output, indent=4)
    with host_file.open("w") as output:
        json.dump(host_result, output, indent=4)

    print("\nGPU selecionada: {} ({})".format(gpu_name, logical_name))
    print(
        "MODEL-ONLY: {:.2f} FPS | {:.4f} ms".format(
            model_result["fps_mean"], model_result["latency_mean_ms"]
        )
    )
    print(
        "HOST-TO-HOST: {:.2f} FPS | {:.4f} ms".format(
            host_result["fps_mean"], host_result["latency_mean_ms"]
        )
    )
    print("Resultados: {} e {}".format(model_file, host_file))


if __name__ == "__main__":
    main()
