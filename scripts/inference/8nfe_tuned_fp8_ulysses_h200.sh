#!/usr/bin/env bash
# 8-NFE, 8 GPUs, inference kernels + fp8 + branch-parallel Ulysses 6+2 (the H200-optimal split).
#   bash scripts/inference/8nfe_tuned_fp8_ulysses_h200.sh
# Renders the Stage-DMD turbo adapter on the VDN hybrid (step 250) on prompts/example_2.pt;
# how it runs is configs/inference/8nfe_tuned_fp8_ulysses_h200.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/inference/infer_ulysses.py \
    --config configs/inference/8nfe_tuned_fp8_ulysses_h200.yaml \
    checkpoint=ckpts/stage-dmd-step-250 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/8nfe_tuned_fp8_ulysses_h200.mp4
