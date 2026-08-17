"""Rebuild TrafficSignNet and import portable weights with TensorFlow 2.12.

This file deliberately uses only APIs and syntax compatible with Python 3.8.6
and TensorFlow 2.12 / Keras 2.12.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model():
    inputs = keras.Input(shape=(32, 32, 3), name="image")

    x = layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv2d_1")(inputs)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.ReLU(name="relu_1")(x)
    x = layers.MaxPooling2D(name="max_pool_1")(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv2d_2")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.ReLU(name="relu_2")(x)
    x = layers.MaxPooling2D(name="max_pool_2")(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2d_3")(x)
    x = layers.BatchNormalization(name="batch_norm_3")(x)
    x = layers.ReLU(name="relu_3")(x)
    x = layers.MaxPooling2D(name="max_pool_3")(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2d_4")(x)
    x = layers.BatchNormalization(name="batch_norm_4")(x)
    x = layers.ReLU(name="relu_4")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(43, name="logits")(x)

    return keras.Model(inputs, outputs, name="TrafficSignNet")


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Rebuild TrafficSignNet under TF 2.12 and load portable weights."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_dir / "modelo" / "TrafficSignNet_FP32_weights.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "modelo_tf212" / "TrafficSignNet_FP32_TF212.h5",
        help="Output .h5 file. Use a path without an extension for SavedModel.",
    )
    return parser.parse_args()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_portable_weights(path):
    with np.load(str(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = []
        for tensor in metadata["tensors"]:
            array = archive[tensor["key"]]
            if list(array.shape) != tensor["shape"]:
                raise ValueError("Corrupt shape for %s" % tensor["key"])
            if str(array.dtype) != tensor["dtype"]:
                raise ValueError("Corrupt dtype for %s" % tensor["key"])
            if array_sha256(array) != tensor["sha256"]:
                raise ValueError("Checksum failed for %s" % tensor["key"])
            arrays.append(array)
    return arrays, metadata


def main():
    args = parse_args()
    print("Python-compatible rebuild using TensorFlow %s" % tf.__version__)
    if not tf.__version__.startswith("2.12."):
        print("WARNING: intended target is TensorFlow 2.12.x")

    model = create_model()
    imported, metadata = load_portable_weights(args.weights)
    expected = model.get_weights()

    if len(imported) != len(expected):
        raise ValueError(
            "Weight count mismatch: archive has %d, model expects %d"
            % (len(imported), len(expected))
        )
    for index, (source, target) in enumerate(zip(imported, expected)):
        if source.shape != target.shape:
            raise ValueError(
                "Weight %d shape mismatch: archive %s, model %s"
                % (index, source.shape, target.shape)
            )

    model.set_weights(imported)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output), include_optimizer=False)

    print("Loaded and verified %d tensors." % metadata["tensor_count"])
    print("Parameters: %d" % model.count_params())
    print("Rebuilt model saved to %s" % args.output)
    model.summary()


if __name__ == "__main__":
    main()
