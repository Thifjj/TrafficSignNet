# TrafficSignNet documentation

This project uses one model and one reproducible data pipeline across training, quantization, host benchmarks, and ZCU104 deployment.

- [Dataset](dataset.md): download, extraction, split layout, and labels.
- [Training and models](training.md): architecture, model files, reconstruction, and quantization.
- [Benchmarks](benchmarks.md): CPU, NVIDIA GPU, metrics, and result locations.
- [ZCU104 deployment](deployment-zcu104.md): compilation, packaging, board execution, and troubleshooting.

The input is RGB `32x32x3`, normalized to `[0, 1]`; the output is 43 logits and the class is selected with `argmax`.
