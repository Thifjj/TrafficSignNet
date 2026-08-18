#!/usr/bin/env python3
import os
import time
import json
from pathlib import Path

# Force CPU before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "keras" / "TrafficSignNet_FP32.h5"
RESULTS_DIR = PROJECT_DIR / "results"

WARMUP = 1000
RUNS = 5
RUN_SECONDS = 60


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

# MODEL-ONLY:
# input is already float32 and normalized/prepared.
@tf.function(
    input_signature=[
        tf.TensorSpec((1, 32, 32, 3), tf.float32)
    ]
)
def model_only(value):
    return model(value, training=False)


# HOST-TO-HOST:
# starts with uint8 input in host memory.
# Includes:
#   uint8 -> float32
#   normalization /255
#   model inference
#   argmax
@tf.function(
    input_signature=[
        tf.TensorSpec((1, 32, 32, 3), tf.uint8)
    ]
)
def host_to_host(value):
    value = tf.cast(value, tf.float32) / 255.0
    logits = model(value, training=False)
    return tf.argmax(logits, axis=1, output_type=tf.int32)


# ============================================================
# INPUTS
# ============================================================

# Already-prepared FP32 input for model-only
float_input = tf.random.stateless_uniform(
    (1, 32, 32, 3),
    seed=(42, 42),
    dtype=tf.float32,
)

# uint8 host input for host-to-host
uint8_input = np.random.default_rng(42).integers(
    0,
    256,
    size=(1, 32, 32, 3),
    dtype=np.uint8,
)


# ============================================================
# BENCHMARK FUNCTION
# ============================================================

def benchmark(name, infer, value):
    print("\n========================================")
    print(name)
    print("========================================")
    print("Batch       : 1")
    print(f"Warmup      : {WARMUP}")
    print(f"Runs        : {RUNS}")
    print(f"Seconds/run : {RUN_SECONDS}")
    print("========================================")

    print("\nWarmup...")

    for _ in range(WARMUP):
        infer(value).numpy()

    runs = []

    for run_id in range(1, RUNS + 1):
        latencies_ns = []

        run_start = time.perf_counter_ns()
        deadline = run_start + int(RUN_SECONDS * 1e9)

        while time.perf_counter_ns() < deadline:
            start = time.perf_counter_ns()

            output = infer(value)
            output.numpy()  # force completion

            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

        run_end = time.perf_counter_ns()
        elapsed = (run_end - run_start) / 1e9

        latency_ms = np.asarray(latencies_ns, dtype=np.float64) / 1e6
        fps = len(latency_ms) / elapsed

        result = {
            "run": run_id,
            "inferences": len(latency_ms),
            "elapsed_seconds": elapsed,
            "fps": float(fps),
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
            f"Run {run_id}: "
            f"{result['fps']:.2f} FPS | "
            f"mean {result['latency_ms']['mean']:.4f} ms | "
            f"P99 {result['latency_ms']['p99']:.4f} ms"
        )

    fps_values = np.asarray([r["fps"] for r in runs])

    def mean_latency(metric):
        return float(np.mean([
            r["latency_ms"][metric] for r in runs
        ]))

    summary = {
        "mode": name,
        "batch_size": 1,
        "warmup": WARMUP,
        "runs": RUNS,
        "seconds_per_run": RUN_SECONDS,
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
    print(f"FPS mean       : {summary['fps_mean']:.2f}")
    print(f"FPS median     : {summary['fps_median']:.2f}")
    print(
        f"FPS min / max  : "
        f"{summary['fps_min']:.2f} / {summary['fps_max']:.2f}"
    )
    print(f"Mean latency   : {summary['latency_mean_ms']:.4f} ms")
    print(f"Median latency : {summary['latency_median_ms']:.4f} ms")
    print(f"P90            : {summary['p90_ms']:.4f} ms")
    print(f"P95            : {summary['p95_ms']:.4f} ms")
    print(f"P99            : {summary['p99_ms']:.4f} ms")
    print(f"P99.9          : {summary['p99_9_ms']:.4f} ms")
    print("----------------------------------------")

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("========================================")
    print("TrafficSignNet CPU Benchmark")
    print("========================================")
    print(f"TensorFlow   : {tf.__version__}")
    print(f"Logical CPUs : {os.cpu_count()}")
    print(f"GPUs visible : {tf.config.list_physical_devices('GPU')}")
    print("========================================")

    # 1) MODEL-ONLY
    model_result = benchmark(
        "MODEL-ONLY",
        model_only,
        float_input,
    )

    # 2) HOST-TO-HOST
    host_result = benchmark(
        "HOST-TO-HOST UINT8",
        host_to_host,
        uint8_input,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_file = RESULTS_DIR / "cpu_model_only.json"
    host_file = RESULTS_DIR / "cpu_host_to_host.json"

    with open(model_file, "w") as f:
        json.dump(model_result, f, indent=4)

    with open(host_file, "w") as f:
        json.dump(host_result, f, indent=4)

    print("\n========================================")
    print("COMPARISON")
    print("========================================")
    print(
        f"MODEL-ONLY   : "
        f"{model_result['fps_mean']:.2f} FPS | "
        f"{model_result['latency_mean_ms']:.4f} ms"
    )
    print(
        f"HOST-TO-HOST: "
        f"{host_result['fps_mean']:.2f} FPS | "
        f"{host_result['latency_mean_ms']:.4f} ms"
    )
    print("========================================")
    print(f"Saved: {model_file}")
    print(f"Saved: {host_file}")


if __name__ == "__main__":
    main()
