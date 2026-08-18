# TrafficSignNet on ZCU104

Self-contained Vitis AI 3.5 deployment bundle for the standard ZCU104
`DPUCZDX8G` platform. It includes the compiled XMODEL and a balanced GTSRB test
subset (10 images from each of the 43 classes).

See `../../DOCUMENTACAO.md` in the repository for the complete project documentation.

## Copy one archive to the board

On the host, generate and copy the archive:

```bash
./scripts/package_zcu104.sh
scp trafficsignnet_zcu104.tar.gz root@ZCU104_IP:/home/root/
```

On the ZCU104:

```bash
cd /home/root
tar -xzf trafficsignnet_zcu104.tar.gz
cd zcu104
./check_board.sh
./run.sh
```

With no arguments, `run.sh` measures accuracy on all bundled images and then
runs five 10-second passes for each of the two benchmark modes. Results are
written to `results.json` under `model_only` and `host_to_host`.

- **model-only** reuses an input already converted to INT8 and measures the VART
  runner/DPU execution and synchronization.
- **host-to-host** starts from an in-memory 32x32 RGB `uint8` image and includes
  normalization, INT8 quantization, DPU execution, synchronization, and
  `argmax`. Image file reading and resize are outside the measurement, matching
  the original host-tensor benchmark convention.

Useful alternatives:

```bash
# Quick smoke test
./run.sh --accuracy --benchmark --warmup 5 --runs 1 --seconds 2

# Accuracy only
./run.sh --accuracy

# Classify one image
./run.sh --image "$(find dataset/00014 -type f | sort | head -n 1)"

# Longer benchmark
./run.sh --benchmark --warmup 1000 --runs 5 --seconds 60
```

The image pipeline is RGB resize to 32x32, conversion to float32 in `[0,1]`,
then INT8 quantization using the input tensor's XMODEL `fix_point`. Both modes
use synchronous batch-1 DPU execution.

The XMODEL must match the DPU fingerprint installed on the board. This bundle
uses the Vitis AI 3.5 standard ZCU104 architecture:

```text
/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
```
