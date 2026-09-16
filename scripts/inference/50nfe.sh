#!/usr/bin/env bash
# 50-NFE, one GPU, released arithmetic (no inference kernels), bf16.
#   bash scripts/inference/50nfe.sh
# Renders the Stage-B c1/VDN/anchor hybrid (step 2000) on prompts/example_2.pt;
# how it runs is configs/inference/50nfe.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."

python src/inference/infer.py \
    --config configs/inference/50nfe.yaml \
    checkpoint=ckpts/stage-b-step-2000 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/50nfe.mp4
