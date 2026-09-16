# vdn-minimax-h3-consumer-gpus

**VDN-MiniMax-H3 video generation on one 24–48 GB GPU. Patched SGLang, one command.**

[VDN-MiniMax-H3](https://github.com/OpenVDN/vdn-minimax-h3) generates 1344×768 video with audio in eight denoising steps. [SGLang](https://github.com/sgl-project/sglang) runs it fastest, but on consumer cards its 8-bit loader crashes while loading the weights, before the first step. This repository is SGLang `d72e595` plus four small patches and three scripts, so the same model runs on an RTX 5090 or RTX PRO 5000 where it could not start, and faster than SGLang's own path on an RTX 4090. Same weights, same eight steps, no retraining.

## Results: one GPU

Two clip lengths: **5.2 s** (124 frames) and **14.4 s** (345 frames), both 1344×768, 24 fps, video + audio. Wall-clock for one generation call through the saved MP4, mean of three runs. Top block: unmodified SGLang. **Bold block: this repo**, on cards where the unmodified loader does not start (✗).

| GPU | Runs on | 5.2 s clip | 14.4 s clip | $ per 14.4 s clip | $ per second of video |
|---|---|---:|---:|---:|---:|
| B300 · 288 GB | SGLang, unmodified | 21 s | 59 s | $0.129 | $0.0090 |
| B200 · 180 GB | SGLang, unmodified | 21 s | 61 s | $0.105 | $0.0073 |
| H100 SXM · 80 GB | SGLang, unmodified | 38 s | 111 s | $0.101 | $0.0071 |
| RTX PRO 6000 · 96 GB | SGLang, unmodified | 85 s | 251 s | $0.155 | $0.0108 |
| L40S · 48 GB | SGLang, unmodified (BF16 offload) | 144 s | 446 s | $0.186 | $0.0129 |
| **RTX PRO 5000 · 48 GB** | **this repo** (SGLang ✗) | **116 s** | **341 s** | **$0.066** | **$0.0046** |
| **RTX 5090 · 32 GB** | **this repo** (SGLang ✗) | **85 s** | **254 s** | **$0.046** | **$0.0032** |
| **RTX 4090 · 24 GB** | **this repo** (SGLang 231 s, BF16 offload) | **194 s** | **✗** | — | **$0.0046** (5.2 s clip) |

- **RTX 4090**: 8-bit against SGLang's BF16 offload on the same host in the same session, 16% faster. The 14.4 s clip fits on no 24 GB path: the first DiT forward of 345 frames alone holds about 22 GiB of activations.
- **L40S**: BF16 offload fits there, and this repo's FP8 path is 14% slower there, so `generate.py` uses unmodified SGLang on the L40S.
- **Repeat on a second machine** (5.2 s / 14.4 s clip): RTX 5090 87 s / 257 s (+3% / +1%), RTX PRO 5000 114 s / 333 s (-2% / -2%). Outputs were stable within each machine; two 5.2 s and two 14.4 s formal outputs on the second RTX 5090 were byte-identical to the first host's, while no 5.2 s and no 14.4 s formal outputs on the second RTX PRO 5000 were byte-identical to the first host's, which ran a different driver version. Hashes are in the results JSON under `cross_host_repeats`.

A second of finished video costs a third of a cent on an RTX 5090, half a cent on an RTX PRO 5000 or RTX 4090, 0.7 cents on an H100 or B200, and 0.9 cents on a B300.

## Results: many GPUs

Ulysses sequence parallelism = GPU count, ring attention 1, eager execution. H100 and B200: unmodified SGLang, FP8 on H100 and MXFP8 on B200, text encoder offloaded on H100. **RTX 5090: this repo** (rank-local loader and main-stream patch, MXFP8, transformer and text encoder offloaded), priced for the GPUs used; every RTX 5090 row ran on an eight-card node. Each cell: time · $ per clip · $ per second of video. **★ = real time**: the clip is generated in less time than it plays.

**5.2 s clip (124 frames)**

| | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---|---:|---:|---:|---:|
| **RTX 5090** (this repo) | 85 s · $0.015 · $0.0030/s | 55 s · $0.020 · $0.0039/s | 32 s · $0.023 · $0.0045/s | 21 s · $0.031 · $0.0060/s |
| H100 SXM | 38 s · $0.035 · $0.0068/s | 21 s · $0.038 · $0.0074/s | 11 s · $0.041 · $0.0079/s | 7.5 s · $0.055 · $0.0106/s |
| B200 | 21 s · $0.037 · $0.0071/s | 12 s · $0.041 · $0.0080/s | 6.5 s · $0.045 · $0.0087/s | **3.8 s · $0.053 · $0.0103/s ★** |

**14.4 s clip (345 frames)**

| | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---|---:|---:|---:|---:|
| **RTX 5090** (this repo) | 254 s · $0.046 · $0.0032/s | 159 s · $0.057 · $0.0040/s | 128 s ‡ · $0.092 · $0.0064/s | 49 s · $0.071 · $0.0049/s |
| H100 SXM | 111 s · $0.101 · $0.0071/s | 60 s † · $0.109 · $0.0076/s | 32 s · $0.115 · $0.0080/s | 19 s · $0.141 · $0.0098/s |
| B200 | 61 s · $0.105 · $0.0073/s | 34 s · $0.119 · $0.0082/s | 18 s · $0.125 · $0.0087/s | **10 s · $0.140 · $0.0098/s ★** |

† provisional timing. Eight GPUs cut latency 5–6× and raise the cost per clip by 30–60%: a 14.4 s clip on eight H100s costs $0.141 and on eight B200s $0.140, against $0.046 on one RTX 5090. On RTX 5090, eight cards bring the 14.4 s clip from 254 s to 49 s at $0.071 per clip, two cards to 159 s at $0.057. The RTX 5090 rows run with the sequence-parallel main-stream patch (see the patch table below): in 4 of 6 cells the three timed requests decoded to identical video and audio; in the other 2 two of the three decoded identically and the audio was identical throughout, the odd one out differing from the other two by an average PSNR of 57 dB or better (per-cell counts in the results JSON). Without the patch, and on RTX PRO 6000 multi-GPU, every repeat differed ([diagnostic](benchmarks/sp-repeatability-diagnostic.json)); the RTX PRO 6000 multi-GPU rows stay withdrawn. ‡ each of the three timed requests in this cell spent 26–56 s between the end of decoding and the saved MP4, a host-side step that took about 2 s in every other cell; the model stages themselves summed to 83 s. Published as measured.

**More memory on one card buys no throughput.** On a B300 (288 GB) one request already saturates the GPU. Asking SGLang for n outputs in one request (`num_outputs_per_prompt`) runs them one after another: 40 s for 2, 80 s for 4, 162 s for 8 short clips against 21 s for one, with peak memory only 107–123 GB; the long clip behaves the same (115 s for 2, 232 s for 4). Two independent processes on the same card each run at half speed (43 s per short clip); a third did not fit. Throughput tops out near 181 short clips or 62 long clips per hour per B300, 8% above one request at a time, and every step of parallelism beyond that only adds waiting. More clips per hour means more cards, not a bigger one. Per-cell numbers: `single_gpu_parallelism` in the [baselines JSON](benchmarks/reproduced-upstream-baselines.json).

## Run

```bash
./install.sh                    # Python 3.12 environment + pinned SGLang, a few minutes
python3 download_models.py      # pinned weights, about 145 GB, into ./models
python3 generate.py --prompt "A hand pours water from a glass pitcher into a tumbler on a wooden table."
```

`generate.py` detects your GPU, picks the recipe we measured for it, applies exactly the patches that recipe needs, generates one 5.2 s clip (124 frames) and prints the MP4 path and the time. Options: `--length long` for a 14.4 s clip (345 frames), `--seed`, `--sound` and `--music` for the audio lines of the prompt, `--gpu` to override detection, `--gpus 2|4|8` on H100, B200 or RTX 5090, `--dry-run` to see the plan. The first request on a machine also compiles SGLang's kernels, so it takes longer than the tables above (121 s instead of 85 s for the 5.2 s clip when we ran the published archive on a fresh RTX 5090 host); the table times are formal requests after two warm-ups. Requirements: Linux, an NVIDIA driver for CUDA 13, the CUDA 13 toolkit (`nvcc` reachable through `CUDA_HOME`, `PATH` or `/usr/local/cuda`; SGLang compiles its kernels at first use), `git`, `ffmpeg`, about 230 GB of free disk (140 GB of weights, a 62 GB copy that SGLang writes under `~/.cache/sgl_diffusion` the first time it loads the model, 10 GB of packages), and a lot of host RAM: the loader stages the checkpoint on the CPU, and every machine we measured on had at least 180 GB.

## What we changed, and why it did not work before

SGLang's 8-bit path builds each full transformer tensor on the GPU and quantizes it there. On 32 GB and 48 GB cards that runs out of memory during loading, so no amount of layer offloading helps: the offload machinery never gets to run. The only route that started on small cards was BF16 with layer offload, which is slow (231 s per 5.2 s clip on an RTX 4090) and streams the full-precision transformer through the GPU on every step.

| Patch | What it does | Needed on |
|---|---|---|
| Streamed quantization loader | Keeps the checkpoint on the CPU and quantizes one module at a time on the GPU, with SGLang's own post-processing code. | RTX 4090, RTX 5090, RTX PRO 5000 |
| Linear-lifetime patch | Frees the temporary buffers inside the transformer's linear layers earlier. | RTX PRO 5000 |
| Rank-local loader | The streamed loader for several GPUs: every rank keeps its own copy of the transformer, quantizes it on its own card, and the ranks load one after another behind a lock. | RTX 5090 × 2, 4, 8 |
| Sequence-parallel main stream | Moves the VDN linear readout off a private CUDA side stream under multi-GPU sequence parallelism; on that side stream, repeated fixed-seed runs gave different outputs. Opt-in. | RTX 5090 × 2, 4, 8 |

Each patch is a checked, idempotent tool in `reproduce/affordable-video-768/tools/`; the runner verifies the SHA-256 of every touched source file before loading the model, so you run exactly the code behind the numbers above. The model is untouched: the 8-bit formats are the ones SGLang selects on each card (FP8 on Ada and Hopper, MXFP8 on Blackwell).

## Where the prices come from

Dollar figures are GPU rental for the measured seconds only: `price per GPU-hour × GPUs × seconds / 3600`, and per second of video divided by the clip length (5.17 s or 14.38 s). The price per GPU-hour is the median on-demand price across cloud providers as listed by [GetDeploying](https://getdeploying.com) on 2026-09-15: $0.44/h RTX 4090 (20 providers), $0.65/h RTX 5090 (20), $0.70/h RTX PRO 5000 (no median published; cheapest in-stock of 4), $1.50/h L40S (35), $2.22/h RTX PRO 6000 (45), $3.29/h H100 (53), $6.25/h B200 (35), $7.87/h B300 (16, retrieved 2026-09-16). Sources per card: [gpu-price-basis.json](benchmarks/gpu-price-basis.json). Excluded: model loading, warmup, idle time, storage, bandwidth. Your provider's price will differ; recompute with your rate and the seconds above. What we actually paid for the machines we rented is recorded per row in the results JSON as `whole_node_hourly_usd`.

## Measurement

SGLang `d72e595`, 8 DiT forwards, eager execution, three formal requests per cell after two warmups, every output probed and fully decoded. The benchmark runner, the per-card recipes and the exact protocol: [reproduce/affordable-video-768](reproduce/affordable-video-768/README.md). Per-run timings, machines rented, precision and offload settings, failed baseline attempts: [results JSON](benchmarks/affordable-video-768-results.json) · [unmodified SGLang runs](benchmarks/reproduced-upstream-baselines.json) · [method](benchmarks/README.md). 8-bit outputs differ from BF16 outputs at the byte level; no cross-hardware quality-equivalence claim is made.

`src/`, `configs/`, `scripts/`, `prompts/` and `diffusers_patches/` are the OpenVDN reference code from the upstream repository (the model definition, the trainers and the native Diffusers inference path); apart from a few trimmed comments they are unchanged. The SGLang route above does not use them.

## Credits and license

Built on [SGLang](https://github.com/sgl-project/sglang), [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3) and [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). Code: [Apache-2.0](LICENSE), see [NOTICE](NOTICE). Weights: [MiniMax H3 Community License](licenses/MiniMax-H3-Community-License-Agreement.txt), which restricts where the model may be used; read it before downloading.
