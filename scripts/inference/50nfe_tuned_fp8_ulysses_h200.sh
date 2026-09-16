#!/usr/bin/env bash
# 50-NFE, 8 GPUs, inference kernels + fp8 + branch-parallel Ulysses 6+2 (the H200-optimal split).
#   bash scripts/inference/50nfe_tuned_fp8_ulysses_h200.sh
# Renders the Stage-B c1/VDN/anchor hybrid (step 2000) on prompts/example_2.pt;
# how it runs is configs/inference/50nfe_tuned_fp8_ulysses_h200.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/inference/infer_ulysses.py \
    --config configs/inference/50nfe_tuned_fp8_ulysses_h200.yaml \
    checkpoint=ckpts/stage-b-step-2000 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/50nfe_tuned_fp8_ulysses_h200.mp4
