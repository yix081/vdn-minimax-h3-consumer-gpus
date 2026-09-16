#!/usr/bin/env bash
# 8-NFE, 8 GPUs, inference kernels + fp8 + branch-parallel Ulysses 5+3 (the B200-optimal split: 1.425 vs 6+2's 1.629 s/NFE there).
#   bash scripts/inference/8nfe_tuned_fp8_ulysses_b200.sh
# Renders the Stage-DMD turbo adapter on the VDN hybrid (step 250) on prompts/example_2.pt;
# how it runs is configs/inference/8nfe_tuned_fp8_ulysses_b200.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/inference/infer_ulysses.py \
    --config configs/inference/8nfe_tuned_fp8_ulysses_b200.yaml \
    checkpoint=ckpts/stage-dmd-step-250 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/8nfe_tuned_fp8_ulysses_b200.mp4
