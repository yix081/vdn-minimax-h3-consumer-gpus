# Reproduce the 1344×768 deployment benchmark

This directory contains the benchmark runner and released H100, B200, and mature single-GPU recipes.

## Fixed workload

- Model: `OpenVDN/vdn-minimax-h3`
- SGLang source: `d72e59508b7554045cb51827f9b8d0f08c7a3abc`
- Output: 1344×768, 24 fps, stereo 32 kHz audio
- Short request: 124 frames, seed 17
- Long request: 345 frames, seed 1000
- Sampling: 9 configured steps and 8 DiT forwards
- Performance protocol: one feasibility request, two warmups, and three formal requests

## Install the measured software snapshot

The recorded environment used Linux, Python 3.12, CUDA 13 compatible NVIDIA drivers, and the package versions in `requirements-frozen.txt`. Create a clean environment and install the exact SGLang revision:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-dev build-essential git ffmpeg pkg-config libnuma1

export REPRO_ROOT="$(pwd)/reproduce/affordable-video-768"
export SGLANG_SOURCE="$(pwd)/sglang-d72e595"

git clone https://github.com/sgl-project/sglang.git "$SGLANG_SOURCE"
git -C "$SGLANG_SOURCE" checkout d72e59508b7554045cb51827f9b8d0f08c7a3abc

python3 -m venv .venv-h3
.venv-h3/bin/pip install uv
.venv-h3/bin/uv pip install --python .venv-h3/bin/python --no-deps \
  -r "$REPRO_ROOT/requirements-frozen.txt"
SGLANG_BUILD_RUST_EXTS=none .venv-h3/bin/uv pip install \
  --python .venv-h3/bin/python --no-deps -e "$SGLANG_SOURCE/python"
```

The frozen file is the complete package snapshot from the measured CUDA 13 environment. Binary wheels must support the local Python, CUDA, and GPU architecture. Do not silently substitute package versions when comparing against the published measurements.

Stage the three model repositories at their measured immutable revisions. This downloads the complete pinned snapshots (about 145 GB; the offline loader rejects a partial snapshot), verifies every runtime file by size and SHA-256, and then writes the offline refs and receipt:

```bash
export HF_HOME="$(pwd)/.cache/h3-models"
.venv-h3/bin/python "$REPRO_ROOT/tools/stage_model.py" --hf-home "$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Use `--use-auth-token` if the repositories require an existing Hugging Face login. For an already staged cache, `--verify-only` rechecks every file without network access or writes. The generation runner requires this receipt, the same `HF_HOME`, both offline flags, and the pinned refs; it does not rehash 140 GB on every invocation.

### Source profiles

The optimized B200 recipes use an exact loaded-projection AdaLN cache. Apply the checked, idempotent patch before running an `*-adaln.json` recipe or the recorded one-GPU cache-off ablation:

```bash
.venv-h3/bin/python "$REPRO_ROOT/tools/apply_b200_adaln_patch.py" \
  --source "$SGLANG_SOURCE" --apply \
  --receipt results/b200-patch-receipt.json
```

The 24/32/48 GB recipes stage quantized transformer weights on the CPU and perform the existing per-module post-processing on the GPU. Apply the streamed-loader patch for RTX 4090 and RTX 5090 short runs:

```bash
.venv-h3/bin/python "$REPRO_ROOT/tools/apply_stream_quant_patch.py" \
  --source "$SGLANG_SOURCE" --apply \
  --receipt results/stream-loader-patch-receipt.json
```

RTX 5090 long installs the lifetime patch but leaves it disabled, matching the measured source state. RTX PRO 5000 enables it. Apply it after the streamed-loader patch for either profile:

```bash
.venv-h3/bin/python "$REPRO_ROOT/tools/apply_linear_lifetime_patch.py" \
  --source "$SGLANG_SOURCE" --apply \
  --receipt results/linear-lifetime-patch-receipt.json
```

Every patch tool refuses an unknown source file and is safe to rerun. The runner checks every patchable source file (and the absence of files a patch would add) before importing the model. Use a separate checkout for each row in the table below; a recipe will reject undeclared patches, including a dormant one.

## Run one recipe

Set `CUDA_VISIBLE_DEVICES` to exactly the number of GPUs in the recipe, then start **one ordinary Python process**. `DiffGenerator.from_pretrained()` starts the SGLang worker processes itself. Do not wrap this command in `torchrun`.

```bash
export REPRO_ROOT="$(pwd)/reproduce/affordable-video-768"
export CUDA_VISIBLE_DEVICES=0,1,2,3

.venv-h3/bin/python "$REPRO_ROOT/generate.py" \
  --output "$(pwd)/results/h100-4-short" \
  --recipe "$REPRO_ROOT/recipes/h100-4.json" \
  --cases "$REPRO_ROOT/cases/short.json" \
  --suite performance
```

Change the recipe and visible device list together for 1, 2, 4, or 8 GPUs. H100 and B200 recipes accept either case file. Single-GPU recipes include `short` or `long` in their name and reject the other case file. The output directory must not exist so an earlier run cannot be overwritten.

The runner records complete request time, resolved runtime arguments, source/configuration hashes, reported peak GPU memory, per-stage metrics when available, media metadata, and a full audio/video decode check. A run is publishable only when `report.json` ends with `"status": "completed"` and every formal output passes validation.
Single-GPU recipes also verify the GPU name, memory floor, and compute capability before loading weights.

## Recipe profiles

| Profile | Recipes | Required source | Role |
|---|---|---|---|
| Upstream baseline | `h100-{1,2,4,8}.json`, `b200-{1,2,4,8}-native.json`, `rtx4090-native-short.json` | Pristine pinned checkout | Reproduced upstream baseline (the RTX 4090 recipe is the BF16 offload path measured next to `rtx4090-short.json`) |
| Cache-off ablation | `b200-1-native-ablation.json` | Patched checkout, cache disabled | Matches the recorded one-GPU B200 row; it is not a pristine-source baseline |
| AdaLN optimization | `b200-{1,2,4,8}-adaln.json` | Patched checkout, cache enabled | Hardware-specific optimization profile |
| Streamed quant loader | `rtx4090-short.json`, `rtx5090-short.json` | Stream-loader patch | Fits quantized weights on 24/32 GB cards |
| Streamed loader, lifetime off | `rtx5090-long.json` | Stream-loader + lifetime patches | Matches the measured 5090 long source state; VAE CPU offload enabled |
| Streamed loader, lifetime on | `rtx-pro5000-{short,long}.json` | Stream-loader + lifetime patches | 48 GB Blackwell profile with shorter temporary lifetimes |
| Pristine single-GPU | `l40s-{short,long}.json`, `rtx-pro6000-{short,long}.json` | Pristine pinned checkout | BF16 L40S or automatic MXFP8 PRO 6000 |
| Rank-local loader + main stream | `rtx5090-{2,4,8}-{short,long}.json` | `tools/apply_ranklocal_loader_patch.py` + `tools/apply_sp_main_stream_patch.py`, `AFFORDABLE_H3_RANKLOCAL_STREAM=1`, `LEANVDN_SP_MAIN_STREAM=1` | Multi-GPU RTX 5090: each rank quantizes its own transformer copy behind a serial lock; the readout stays on the current stream |
| Sequence-parallel main stream | (used by the rows above) | `tools/apply_sp_main_stream_patch.py` + `LEANVDN_SP_MAIN_STREAM=1` | Moves the VDN linear readout off its private side stream; repeated fixed-seed two-GPU runs then decode byte-identically far more often (per-cell counts in `benchmarks/affordable-video-768-results.json`). Timings are in the top-level README |

Each recipe declares `source_profile`; the runner rejects a pristine, dormant-patch, or active-cache mismatch. Published results must use the profile recorded with that result.

The native B200 recipes preserve the archived server arguments, including the server-side 345-frame, 1344×768 warmup. The historical long-video baseline used two client warmups followed by three formal requests for 1/2/4 GPUs. Its eight-GPU value pools two cohorts with five and three formal requests (`n=8`). This runner uses the release protocol of one feasibility request, two client warmups, and three formal requests per invocation. A new `n=3` run therefore validates the same configuration but does not recreate the historical pooled sample count.

The RTX 5090, L40S, RTX PRO 5000, and RTX 4090 cohorts used two warmups and three formal requests without a separate feasibility request; their recipes preserve that protocol. The RTX PRO 6000 cohorts used one feasibility request, two warmups, and three formal requests. `rtx4090-short.json` and `rtx4090-native-short.json` are the only released RTX 4090 recipes: every 345-frame attempt on 24 GB, with either source profile, ran out of GPU memory in the first DiT forward.
