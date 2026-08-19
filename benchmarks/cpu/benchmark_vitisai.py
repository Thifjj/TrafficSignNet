#!/usr/bin/env python3
"""Benchmark the Vitis AI quantized TrafficSignNet model on CPU.

Compatible with Python 3.8 and TensorFlow 2.12. The Vitis AI TensorFlow
quantizer package must be installed in the environment used to run it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force CPU execution before TensorFlow/Vitis AI are imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

try:
    from tensorflow_model_optimization.quantization.keras import vitis_quantize
except ImportError as exc:
    raise ImportError(
        "Vitis AI quantizer is required to load TrafficSignNet_INT8.h5. "
        "Run this script from the same Vitis AI TensorFlow environment used "
        "to create the quantized model."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "keras" / "TrafficSignNet_INT8.h5"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "cpu" / "vitisai"
INPUT_SHAPE = (1, 32, 32, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the Vitis AI INT8 TrafficSignNet model on CPU."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be at least 0")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0")
    return args


def load_quantized_model(model_path):
    if not model_path.is_file():
        raise FileNotFoundError("Quantized model not found: {}".format(model_path))

    # Vitis quantized layers are custom Keras objects and must be restored
    # inside quantize_scope.
    with vitis_quantize.quantize_scope():
        return tf.keras.models.load_model(str(model_path), compile=False)


def make_inference_functions(model):
    # The Vitis quantized Keras model still receives normalized float32 data;
    # its input quantizer performs the simulated INT8 conversion internally.
    @tf.function(
        input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)]
    )
    def model_only(value):
        return model(value, training=False)

    @tf.function(
        input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.uint8)]
    )
    def host_to_host(value):
        value = tf.cast(value, tf.float32) / 255.0
        logits = model(value, training=False)
        return tf.argmax(logits, axis=1, output_type=tf.int32)

    return model_only, host_to_host


def benchmark(name, infer, value, warmup, runs_count, run_seconds):
    print("\n========================================")
    print(name)
    print("========================================")
    print("Batch       : 1")
    print("Warmup      : {}".format(warmup))
    print("Runs        : {}".format(runs_count))
    print("Seconds/run : {}".format(run_seconds))
    print("========================================")
    print("\nWarmup...")

    for _ in range(warmup):
        infer(value).numpy()

    runs = []
    for run_id in range(1, runs_count + 1):
        latencies_ns = []
        run_start = time.perf_counter_ns()
        deadline = run_start + int(run_seconds * 1e9)

        while time.perf_counter_ns() < deadline:
            start = time.perf_counter_ns()
            infer(value).numpy()  # Materialize the result and force completion.
            latencies_ns.append(time.perf_counter_ns() - start)

        elapsed = (time.perf_counter_ns() - run_start) / 1e9
        latency_ms = np.asarray(latencies_ns, dtype=np.float64) / 1e6
        result = {
            "run": run_id,
            "inferences": len(latency_ms),
            "elapsed_seconds": elapsed,
            "fps": float(len(latency_ms) / elapsed),
            "latency_ms": {
                "mean": float(latency_ms.mean()),
                "median": float(np.median(latency_ms)),
                "std": float(latency_ms.std()),
                "min": float(latency_ms.min()),
                "max": float(latency_ms.max()),
                "p90": float(np.percentile(latency_ms, 90)),
                "p95": float(np.percentile(latency_ms, 95)),
                "p99": float(np.percentile(latency_ms, 99)),
                "p99_9": float(np.percentile(latency_ms, 99.9)),
            },
        }
        runs.append(result)
        print(
            "Run {}: {:.2f} FPS | mean {:.4f} ms | P99 {:.4f} ms".format(
                run_id,
                result["fps"],
                result["latency_ms"]["mean"],
                result["latency_ms"]["p99"],
            )
        )

    fps_values = np.asarray([run["fps"] for run in runs])

    def mean_latency(metric):
        return float(np.mean([run["latency_ms"][metric] for run in runs]))

    summary = {
        "mode": name,
        "model_format": "Vitis AI quantized Keras H5",
        "batch_size": 1,
        "warmup": warmup,
        "runs": runs_count,
        "seconds_per_run": run_seconds,
        "fps_mean": float(fps_values.mean()),
        "fps_median": float(np.median(fps_values)),
        "fps_min": float(fps_values.min()),
        "fps_max": float(fps_values.max()),
        "latency_mean_ms": mean_latency("mean"),
        "latency_median_ms": mean_latency("median"),
        "p90_ms": mean_latency("p90"),
        "p95_ms": mean_latency("p95"),
        "p99_ms": mean_latency("p99"),
        "p99_9_ms": mean_latency("p99_9"),
        "runs_data": runs,
    }

    print("\n----------------------------------------")
    print("FINAL RESULT")
    print("----------------------------------------")
    print("FPS mean       : {:.2f}".format(summary["fps_mean"]))
    print("FPS median     : {:.2f}".format(summary["fps_median"]))
    print(
        "FPS min / max  : {:.2f} / {:.2f}".format(
            summary["fps_min"], summary["fps_max"]
        )
    )
    print("Mean latency   : {:.4f} ms".format(summary["latency_mean_ms"]))
    print("Median latency : {:.4f} ms".format(summary["latency_median_ms"]))
    print("P90            : {:.4f} ms".format(summary["p90_ms"]))
    print("P95            : {:.4f} ms".format(summary["p95_ms"]))
    print("P99            : {:.4f} ms".format(summary["p99_ms"]))
    print("P99.9          : {:.4f} ms".format(summary["p99_9_ms"]))
    print("----------------------------------------")
    return summary


def main():
    args = parse_args()
    model_path = args.model.resolve()
    results_dir = args.results_dir.resolve()

    print("========================================")
    print("TrafficSignNet Vitis AI CPU Benchmark")
    print("========================================")
    print("Python       : {}".format(sys.version.split()[0]))
    print("TensorFlow   : {}".format(tf.__version__))
    print("Model        : {}".format(model_path))
    print("Logical CPUs : {}".format(os.cpu_count()))
    print("GPUs visible : {}".format(tf.config.list_physical_devices("GPU")))
    print("========================================")

    model = load_quantized_model(model_path)
    model_only, host_to_host = make_inference_functions(model)

    float_input = tf.random.stateless_uniform(
        INPUT_SHAPE, seed=(42, 42), dtype=tf.float32
    )
    uint8_input = np.random.default_rng(42).integers(
        0, 256, size=INPUT_SHAPE, dtype=np.uint8
    )

    model_result = benchmark(
        "VITIS-AI MODEL-ONLY",
        model_only,
        float_input,
        args.warmup,
        args.runs,
        args.seconds,
    )
    host_result = benchmark(
        "VITIS-AI HOST-TO-HOST UINT8",
        host_to_host,
        uint8_input,
        args.warmup,
        args.runs,
        args.seconds,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    model_file = results_dir / "cpu_model_only.json"
    host_file = results_dir / "cpu_host_to_host.json"
    with model_file.open("w") as output_file:
        json.dump(model_result, output_file, indent=4)
    with host_file.open("w") as output_file:
        json.dump(host_result, output_file, indent=4)

    print("\n========================================")
    print("COMPARISON")
    print("========================================")
    print(
        "MODEL-ONLY   : {:.2f} FPS | {:.4f} ms".format(
            model_result["fps_mean"], model_result["latency_mean_ms"]
        )
    )
    print(
        "HOST-TO-HOST: {:.2f} FPS | {:.4f} ms".format(
            host_result["fps_mean"], host_result["latency_mean_ms"]
        )
    )
    print("========================================")
    print("Saved: {}".format(model_file))
    print("Saved: {}".format(host_file))


if __name__ == "__main__":
    main()
