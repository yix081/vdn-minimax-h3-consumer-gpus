#!/usr/bin/env bash
# Stage B: LoRA distillation on the frozen base, teacher + data loss.
#   bash scripts/training/stage_b.sh data.index_file=/path/to/video_index.jsonl [more.dotlist=overrides]
# Starts from ckpts/train/stage_a2_c1_vdn_anchor/hybrid_step000500.pt (see the yaml).
# Our run: 8 nodes x 8 GPUs; step 2000 is the released ckpts/stage-b-step-2000.
# Multi-node: replace --standalone with your scheduler's torchrun rendezvous flags.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/training/train_stage_b.py \
    --config configs/training/stage_b_c1_vdn_anchor.yaml \
    "$@"
