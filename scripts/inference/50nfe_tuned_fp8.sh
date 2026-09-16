#!/usr/bin/env bash
# 50-NFE, one GPU, inference kernels + fp8 (2 warm-up NFE).
#   bash scripts/inference/50nfe_tuned_fp8.sh
# Renders the Stage-B c1/VDN/anchor hybrid (step 2000) on prompts/example_2.pt;
# how it runs is configs/inference/50nfe_tuned_fp8.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."

python src/inference/infer.py \
    --config configs/inference/50nfe_tuned_fp8.yaml \
    checkpoint=ckpts/stage-b-step-2000 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/50nfe_tuned_fp8.mp4
