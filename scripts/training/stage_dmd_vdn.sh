#!/usr/bin/env bash
# Stage DMD, VDN-teacher branch (the released one): DMD2 (no GAN) trains the 8-step
# `turbo` LoRA against the frozen Stage-B VDN's own 50-step teacher.
#   bash scripts/training/stage_dmd_vdn.sh data.index_file=/path/to/video_index.jsonl [more.dotlist=overrides]
# Starts from the released ckpts/stage-b-step-2000 plus an external few-step LoRA
# (turbo.checkpoint in the yaml; see "Stage-DMD" in the README).
# Our run: 4 nodes x 8 GPUs; step 250 is the released ckpts/stage-dmd-step-250.
# Multi-node: replace --standalone with your scheduler's torchrun rendezvous flags.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/training/train_stage_dmd.py \
    --config configs/training/stage_dmd_vdn.yaml \
    "$@"
