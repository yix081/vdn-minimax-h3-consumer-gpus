#!/usr/bin/env bash
# 8-NFE, one GPU, inference kernels + fp8 (2 warm-up NFE).
#   bash scripts/inference/8nfe_tuned_fp8.sh
# Renders the Stage-DMD turbo adapter on the VDN hybrid (step 250) on prompts/example_2.pt;
# how it runs is configs/inference/8nfe_tuned_fp8.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."

python src/inference/infer.py \
    --config configs/inference/8nfe_tuned_fp8.yaml \
    checkpoint=ckpts/stage-dmd-step-250 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/8nfe_tuned_fp8.mp4
