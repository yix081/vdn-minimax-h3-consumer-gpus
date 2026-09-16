#!/usr/bin/env bash
# 50-NFE, 8 GPUs, inference kernels + fp8 + branch-parallel Ulysses 5+3 (the B200-optimal split: 1.425 vs 6+2's 1.629 s/NFE there).
#   bash scripts/inference/50nfe_tuned_fp8_ulysses_b200.sh
# Renders the Stage-B c1/VDN/anchor hybrid (step 2000) on prompts/example_2.pt;
# how it runs is configs/inference/50nfe_tuned_fp8_ulysses_b200.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/inference/infer_ulysses.py \
    --config configs/inference/50nfe_tuned_fp8_ulysses_b200.yaml \
    checkpoint=ckpts/stage-b-step-2000 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/50nfe_tuned_fp8_ulysses_b200.mp4
