# Docker usage

Prebuilt image (once published):
```bash
docker pull <registry>/flux-amd-rocm:7.1
```

Build locally:
```bash
cd flux-amd-rocm
docker build -t flux-amd-rocm:7.1 -f docker/Dockerfile.rocm7.1 .
```

## Run

The container needs access to `/dev/kfd` and `/dev/dri` and must be in the `video` group:

```bash
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --ipc=host --shm-size=16g \
    -e HF_TOKEN=$HF_TOKEN \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/outputs:/workspace/outputs \
    flux-amd-rocm:7.1 \
    python generate.py "a dragon coiled around a tower at sunset" --out outputs/dragon.png
```

The two `-v` mounts are important:
- `~/.cache/huggingface` — persists the 33 GB of FLUX.1-dev weights across runs
- `outputs/` — where the generated PNG ends up on your host

## Different GPU arch

Override the env:
```bash
-e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
-e PYTORCH_ROCM_ARCH=gfx1100 \
```
for RX 7900 series. For RDNA2 (gfx1030) the base ROCm image may require a different tag; check [hub.docker.com/r/rocm/pytorch](https://hub.docker.com/r/rocm/pytorch).

## Multi-GPU

Add more GPUs via `--device`:
```bash
--device=/dev/dri/renderD128 \
--device=/dev/dri/renderD129 \
--device=/dev/dri/renderD130 \
```
(Device numbers depend on your host; check with `ls /dev/dri/`.)

Or mount them all:
```bash
--device=/dev/dri \
```
and set `HIP_VISIBLE_DEVICES=0,1,2,3,4` inside the container.

Multi-GPU support is experimental in this repo (no balanced device_map in the first release). Expect single-GPU performance from one device at a time unless you add your own sharding.
