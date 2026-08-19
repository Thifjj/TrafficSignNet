# ZCU104 deployment

The board bundle is under `deploy/zcu104/`. The canonical compiled model is `models/compiled/zcu104/trafficsignnet_int8.xmodel`.

## Compile and package

Inside the Vitis AI 3.5 TensorFlow 2 environment:

```bash
mkdir -p models/compiled/zcu104
vai_c_tensorflow2 --model models/keras/TrafficSignNet_INT8.h5 \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  --output_dir models/compiled/zcu104 \
  --net_name trafficsignnet_int8
./scripts/package_zcu104.sh
```

The package is written to `artifacts/zcu104/trafficsignnet_zcu104.tar.gz`. Copy it to the board, extract it, run `check_board.sh`, and then run `run.sh`.

```bash
scp artifacts/zcu104/trafficsignnet_zcu104.tar.gz root@ZCU104_IP:/home/root/
ssh root@ZCU104_IP
cd /home/root && tar -xzf trafficsignnet_zcu104.tar.gz
cd zcu104 && ./check_board.sh
./run.sh --accuracy --benchmark --warmup 5 --runs 1 --seconds 2
```

Other useful commands are `./run.sh --accuracy`, `./run.sh --benchmark --warmup 1000 --runs 5 --seconds 60`, and `./run.sh --image path/to/image.png`.

The board code discovers tensor shapes and `fix_point` values from the XMODEL. The current model uses input `[1,32,32,3]`, output `[1,43]`, input `fix_point=6`, and output `fix_point=1`. The XMODEL fingerprint must match the board DPU.
