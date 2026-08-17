#!/usr/bin/env python3
"""TrafficSignNet inference, accuracy, and benchmark for a Vitis AI ZCU104."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import vart
    import xir
except ImportError as exc:
    raise SystemExit(
        "VART Python modules were not found. Run this on the ZCU104 Vitis AI image."
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "trafficsignnet_int8.xmodel"
DEFAULT_DATASET = ROOT / "dataset"
IMAGE_SUFFIXES = {".png", ".ppm", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image", type=Path, help="Classify one image")
    parser.add_argument("--accuracy", action="store_true", help="Evaluate labeled dataset")
    parser.add_argument("--benchmark", action="store_true", help="Measure DPU latency and FPS")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--results", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    if not (args.image or args.accuracy or args.benchmark):
        args.accuracy = True
        args.benchmark = True
    if args.warmup < 0 or args.runs < 1 or args.seconds <= 0:
        parser.error("warmup >= 0, runs >= 1, and seconds > 0 are required")
    return args


def dpu_subgraphs(graph):
    root = graph.get_root_subgraph()
    found = []

    def visit(subgraph):
        if subgraph.has_attr("device") and subgraph.get_attr("device").upper() == "DPU":
            found.append(subgraph)
            return
        for child in subgraph.toposort_child_subgraph():
            visit(child)

    visit(root)
    return found


def tensor_fix_point(tensor):
    return int(tensor.get_attr("fix_point")) if tensor.has_attr("fix_point") else 0


class DpuModel:
    def __init__(self, model_path):
        if not model_path.is_file():
            raise FileNotFoundError("XMODEL not found: {}".format(model_path))
        self.graph = xir.Graph.deserialize(str(model_path))
        subgraphs = dpu_subgraphs(self.graph)
        if len(subgraphs) != 1:
            raise RuntimeError(
                "Expected exactly one DPU subgraph, found {}. The model may contain "
                "unsupported CPU partitions.".format(len(subgraphs))
            )
        self.runner = vart.Runner.create_runner(subgraphs[0], "run")
        inputs = self.runner.get_input_tensors()
        outputs = self.runner.get_output_tensors()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Expected one input and one output tensor")
        self.input_tensor = inputs[0]
        self.output_tensor = outputs[0]
        self.input_shape = tuple(int(x) for x in self.input_tensor.dims)
        self.output_shape = tuple(int(x) for x in self.output_tensor.dims)
        self.input_fix = tensor_fix_point(self.input_tensor)
        self.output_fix = tensor_fix_point(self.output_tensor)
        self.input = np.empty(self.input_shape, dtype=np.int8, order="C")
        self.output = np.empty(self.output_shape, dtype=np.int8, order="C")

        if len(self.input_shape) != 4 or self.input_shape[0] != 1:
            raise RuntimeError("Expected batch-1 NHWC input, got {}".format(self.input_shape))

    def load_host_image(self, image_path):
        """Load an image as the benchmark's host-side uint8 NHWC tensor."""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image: {}".format(image_path))
        height, width = self.input_shape[1], self.input_shape[2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image, dtype=np.uint8)

    def prepare_host_input(self, host_image):
        """Normalize and quantize one already decoded uint8 host image."""
        normalized = host_image.astype(np.float32) / 255.0
        quantized = np.rint(normalized * (2.0 ** self.input_fix))
        self.input[0] = np.clip(quantized, -128, 127).astype(np.int8)
        return self.input

    def prepare(self, image_path):
        return self.prepare_host_input(self.load_host_image(image_path))

    def execute_raw(self):
        job_id = self.runner.execute_async([self.input], [self.output])
        self.runner.wait(job_id)
        return self.output

    def execute(self):
        self.execute_raw()
        return self.output.astype(np.float32) * (2.0 ** (-self.output_fix))

    def host_to_host(self, host_image):
        self.prepare_host_input(host_image)
        output = self.execute_raw()
        return int(np.argmax(output[0]))

    def predict(self, image_path):
        self.prepare(image_path)
        logits = self.execute()
        return int(np.argmax(logits[0])), logits[0]

    def describe(self):
        return {
            "input_name": self.input_tensor.name,
            "input_shape": self.input_shape,
            "input_fix_point": self.input_fix,
            "output_name": self.output_tensor.name,
            "output_shape": self.output_shape,
            "output_fix_point": self.output_fix,
        }


def dataset_images(dataset):
    samples = []
    for class_dir in sorted(dataset.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        label = int(class_dir.name)
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, label))
    if not samples:
        raise RuntimeError("No labeled images found below {}".format(dataset))
    return samples


def evaluate(model, dataset):
    samples = dataset_images(dataset)
    correct = 0
    started = time.perf_counter()
    for index, (path, expected) in enumerate(samples, 1):
        predicted, _ = model.predict(path)
        correct += int(predicted == expected)
        if index % 100 == 0 or index == len(samples):
            print("Accuracy progress: {}/{}".format(index, len(samples)))
    elapsed = time.perf_counter() - started
    result = {
        "images": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples),
        "elapsed_seconds": elapsed,
        "images_per_second": len(samples) / elapsed,
    }
    print("Accuracy: {:.2%} ({}/{})".format(result["accuracy"], correct, len(samples)))
    return result, samples[0][0]


def benchmark(name, scope, infer, warmup, runs, seconds):
    print("\n========================================")
    print(name)
    print("========================================")
    for _ in range(warmup):
        infer()

    run_data = []
    for run_number in range(1, runs + 1):
        latencies = []
        started = time.perf_counter()
        deadline = started + seconds
        while time.perf_counter() < deadline:
            before = time.perf_counter_ns()
            infer()
            latencies.append((time.perf_counter_ns() - before) / 1e6)
        elapsed = time.perf_counter() - started
        values = np.asarray(latencies, dtype=np.float64)
        item = {
            "run": run_number,
            "inferences": len(latencies),
            "elapsed_seconds": elapsed,
            "fps": len(latencies) / elapsed,
            "latency_ms": {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "p99_9": float(np.percentile(values, 99.9)),
            },
        }
        run_data.append(item)
        print(
            "Run {}: {:.2f} FPS, mean {:.4f} ms, P99 {:.4f} ms".format(
                run_number, item["fps"], item["latency_ms"]["mean"],
                item["latency_ms"]["p99"]
            )
        )
    return {
        "mode": name,
        "scope": scope,
        "batch_size": 1,
        "warmup": warmup,
        "runs": runs,
        "seconds_per_run": seconds,
        "fps_mean": float(np.mean([x["fps"] for x in run_data])),
        "latency_mean_ms": float(np.mean([x["latency_ms"]["mean"] for x in run_data])),
        "runs_data": run_data,
    }


def benchmark_both_modes(model, image_path, warmup, runs, seconds):
    host_image = model.load_host_image(image_path)

    # MODEL-ONLY receives an already normalized/quantized INT8 tensor. It
    # measures only submission, DPU execution, synchronization, and output.
    model.prepare_host_input(host_image)
    model_only = benchmark(
        "DPU MODEL-ONLY",
        "Prequantized INT8 input through the VART DPU runner; host preprocessing "
        "and argmax excluded",
        model.execute_raw,
        warmup,
        runs,
        seconds,
    )

    # HOST-TO-HOST starts at an in-memory uint8 image, matching the TensorFlow
    # benchmark: normalization/quantization + model + argmax. File I/O and
    # resize are excluded because the host tensor is loaded once beforehand.
    host_to_host = benchmark(
        "DPU HOST-TO-HOST UINT8",
        "In-memory uint8 input; float normalization, INT8 quantization, VART DPU "
        "execution, synchronization, and argmax included; file I/O/resize excluded",
        lambda: model.host_to_host(host_image),
        warmup,
        runs,
        seconds,
    )

    print("\n========================================")
    print("COMPARISON")
    print("========================================")
    print(
        "MODEL-ONLY   : {:.2f} FPS | {:.4f} ms".format(
            model_only["fps_mean"], model_only["latency_mean_ms"]
        )
    )
    print(
        "HOST-TO-HOST: {:.2f} FPS | {:.4f} ms".format(
            host_to_host["fps_mean"], host_to_host["latency_mean_ms"]
        )
    )
    return model_only, host_to_host


def main():
    args = parse_args()
    model = DpuModel(args.model.resolve())
    results = {"python": sys.version.split()[0], "model": model.describe()}
    print(json.dumps(results["model"], indent=2))

    benchmark_image = args.image.resolve() if args.image else None
    if args.image:
        predicted, logits = model.predict(benchmark_image)
        top5 = np.argsort(logits)[-5:][::-1]
        print("Prediction: class {:05d}".format(predicted))
        print("Top 5: {}".format(
            ", ".join("{:05d} ({:.5f})".format(int(i), float(logits[i])) for i in top5)
        ))
        results["single_image"] = {"path": str(args.image), "prediction": predicted}

    if args.accuracy:
        results["accuracy"], first_image = evaluate(model, args.dataset.resolve())
        benchmark_image = benchmark_image or first_image

    if args.benchmark:
        if benchmark_image is None:
            benchmark_image = dataset_images(args.dataset.resolve())[0][0]
        results["model_only"], results["host_to_host"] = benchmark_both_modes(
            model, benchmark_image, args.warmup, args.runs, args.seconds
        )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w") as output:
        json.dump(results, output, indent=2)
    print("Results saved to {}".format(args.results))


if __name__ == "__main__":
    main()
