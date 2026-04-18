#!/usr/bin/env bash
# Reproduce the canonical "cat sipping a margarita" image used in the reference benchmark.
set -e
cd "$(dirname "$0")/.."
source .env.rocm
python generate.py \
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain" \
    --out outputs/cat_margarita.png \
    --seed 42 --steps 28 --size 1024
