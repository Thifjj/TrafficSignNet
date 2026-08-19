# Training and model artifacts

## Architecture

The model accepts `1x32x32x3` RGB images and produces `1x43` logits:

```text
Conv2D(16) + BatchNorm + ReLU + MaxPool
Conv2D(32) + BatchNorm + ReLU + MaxPool
Conv2D(64) + BatchNorm + ReLU + MaxPool
Conv2D(64) + BatchNorm + ReLU
Flatten -> Dense(64, ReLU) -> Dense(43 logits)
```

There is no softmax because `argmax(logits)` gives the same class.

## Artifacts

- `models/keras/TrafficSignNet_FP32.h5`: original FP32 model.
- `models/keras/TrafficSignNet_FP32_weights.npz`: portable weights and metadata.
- `models/keras/TrafficSignNet_FP32_TF212.h5`: TensorFlow 2.12-compatible reconstruction.
- `models/keras/TrafficSignNet_INT8.h5`: Vitis AI quantized Keras model for host execution.
- `models/compiled/zcu104/trafficsignnet_int8.xmodel`: XMODEL compiled for the standard ZCU104 DPU.

## Reconstruction and quantization

```bash
python3 training/rebuild_model_tf212.py
python3 optimization/quantize_vai.py
```

Quantization uses 32 batches of 32 validation images and the same `/255.0` normalization used during inference. Run these commands inside the compatible TensorFlow/Vitis AI environment.
