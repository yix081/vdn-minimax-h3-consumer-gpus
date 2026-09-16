#!/usr/bin/env bash
# Stage A1: build the hybrid from the dense base and align each layer to its teacher.
#   bash scripts/training/stage_a1.sh data.index_file=/path/to/video_index.jsonl [more.dotlist=overrides]
# Our run: 1 node x 8 GPUs, 200 steps. Writes ckpts/train/stage_a1_c1_vdn_anchor/.
# Multi-node: replace --standalone with your scheduler's torchrun rendezvous flags.
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nproc_per_node=8 src/training/train_stage_a1.py \
    --config configs/training/stage_a1_c1_vdn_anchor.yaml \
    "$@"
