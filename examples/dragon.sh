#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .env.rocm
python generate.py \
    "a majestic red dragon coiled around a medieval stone tower at sunset, detailed scales, golden hour light, cinematic wide shot, highly detailed, fantasy matte painting, volumetric atmosphere" \
    --out outputs/dragon.png \
    --seed 1337 --steps 28 --size 1024
