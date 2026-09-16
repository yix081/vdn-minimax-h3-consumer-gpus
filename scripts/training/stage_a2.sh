#!/usr/bin/env bash
# Stage A2: end-to-end training of the A1 hybrid (FSDP2), full-sequence loss.
#   bash scripts/training/stage_a2.sh data.index_file=/path/to/video_index.jsonl [more.dotlist=overrides]
# Starts from ckpts/train/stage_a1_c1_vdn_anchor/hybrid_step000200.pt (see the yaml).
# Our run: 8 nodes x 8 GPUs, 500 steps. Writes ckpts/train/stage_a2_c1_vdn_anchor/.
# Multi-node: replace --standalone with your scheduler's torchrun rendezvous flags.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/training/train_stage_a2.py \
    --config configs/training/stage_a2_c1_vdn_anchor.yaml \
    "$@"
