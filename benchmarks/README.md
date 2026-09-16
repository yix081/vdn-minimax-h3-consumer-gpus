# Measurement details

## Protocol

VDN-H3 / MiniMax H3, 1344×768, 24 fps, stereo 32 kHz audio. The short request uses 124 frames / seed 17; the long request uses 345 frames / seed 1000. Nine configured sigma points produce eight DiT forwards. A normal new performance cell contains one feasibility request, two warmups, and three formal requests; historical pooled cells may have more formal observations, as recorded in the JSON.

The timer covers the host generation call through the finalized MP4. Model loading, setup, warmup, and post-run media validation are excluded. Completed outputs are checked for video/audio format and full decode. These checks do not establish perceptual quality equivalence.

## Costs

`cost per video = rented node hourly rate × mean generation seconds / 3600`.

The rates are historical prices from the measured rentals. Costs exclude setup, loading, warmup, idle time, storage, and bandwidth. Every completed JSON row records `whole_node_hourly_usd`, `rented_gpu_count`, and `usd_per_clip`; `gpus` is the number used by the request.

- H100 1/2/4-GPU runs rented a four-GPU node; H100 8-GPU runs rented eight.
- RTX PRO 6000 1/2/4/8-GPU runs rented an eight-GPU node.
- RTX 5090 1-GPU runs rented one GPU; 2/4/8-GPU runs rented eight.
- B300 1-GPU runs rented one GPU of an eight-GPU node at its single-GPU rate.
- B200 1-GPU long runs rented one GPU. The AdaLN 2-GPU short run rented two; the unmodified 1/2/4/8-GPU short runs and the 2/4/8-GPU long runs rented eight.
- RTX 4090, L40S, and RTX PRO 5000 successful single-GPU measurements use their recorded single-GPU rental rates.

A subset run on an eight-GPU node is not a dedicated one-, two-, or four-GPU price. Failed and unmeasured cells have no cost-per-completed-video estimate.

## Precision and configuration

B300, B200, RTX 5090, and Blackwell workstation rows use MXFP8; H100 and the RTX 4090 deployment row use FP8; L40S uses BF16. Attention kernels, host memory, offload policy, and power limits also differ. The JSON records each hardware configuration. The main deployment table is not an isolated architecture benchmark or a controlled precision comparison.

## Reproduced upstream baselines

The first rows in the main table are our own runs of unmodified SGLang `d72e59508b7554045cb51827f9b8d0f08c7a3abc`. They use the recorded pouring-water prompt, not an assumed match to an upstream announcement. [Baseline records](reproduced-upstream-baselines.json) include the model revisions, request parameters, precision, parallelism, sample counts and timings.

H100 uses explicit FP8 and text-encoder layerwise offload. B200 and B300 use upstream automatic MXFP8 with the same server arguments (the B200 single-GPU recipe; B300 measured 2026-09-16 on a rented single card, compute capability 10.3, where PyTorch runs its sm_100 kernels). L40S uses native BF16 with transformer/text-encoder layerwise offload. PRO 6000 uses native MXFP8 with text offload, native VAE offload for long requests, and a 400 W per-GPU power limit. These native paths are upstream baselines, not model or kernel changes from this repository. These profiles use eager execution; the exact parallelism and encoder settings are retained per result. No model/kernel patches are enabled in these baseline rows. Short and long inputs use the seeds and frame counts listed above; neither is a denoising-only timing.

The results JSON also carries B200 rows from a separate run with the AdaLN cache patch installed but disabled (one GPU) and enabled (two GPUs, short clip). They are ablations of that patch, not substitutes for the upstream control, and are not shown in the top-level README. The baseline B200 1/2/4/8 long runs all rented the same eight-GPU node; its whole-node charge explains their cost differences from newer smaller-node rentals.

## One GPU, more than one clip at a time

`single_gpu_parallelism` in the baselines JSON records a B300 session that asked for several outputs per request (`num_outputs_per_prompt` 2/4/8 short, 2/4 long) and ran two independent processes on one card. In this SGLang revision the outputs of one request are generated one after another (the denoising stage per output does not change and the request time is linear in the count), and two processes halve each other's speed; a third process ran out of memory during the load-time warmup. Those cells used a one-line change to the runner that reads `num_outputs_per_prompt` from an environment variable and accepts several results; the shipped runner keeps the value at 1, so the cells are recorded as evidence, not as a recipe.

## Evidence status

Every row marked `complete` has three formal timings and fully decoded outputs behind it. Rows marked `provisional` (the H100 two-GPU long clip) were measured once with an unresolved repeatability question; rows marked `failed` (the RTX 4090 long clip, the unmodified loader on RTX 5090 and RTX PRO 5000) record the error that stopped them. Multi-GPU RTX 5090 rows were measured with the sequence-parallel main-stream patch on one eight-card node (2, 4 and 8 cards used, eight billed); each row records whether its three timed outputs decoded identically (`fixed_seed_decoded_outputs_identical`). Multi-GPU RTX PRO 6000 timings are not published because their fixed-seed outputs differed between repeats; A100 was not measured.

## Second-host repeats and quality samples

`cross_host_repeats` in the [results JSON](affordable-video-768-results.json) records one repeat of each RTX 5090 and RTX PRO 5000 row on a second machine of the same GPU type: same recipe, pinned source and model revisions, two warmups and three formal requests. Each row keeps the three formal timings, the relative difference from the published row, and the SHA-256 of every output next to the first host's formal output, so byte-level agreement across machines can be checked. `quality_samples` lists the twelve quality prompts per clip length (six categories, two seeds) generated on each second host with the runner's quality mode; the media are archived, not distributed, and no perceptual metric is published.

The RTX 4090 rows come from one host in one session: this repository's FP8 recipe and the unmodified BF16 offload baseline ran back to back on the same machine. Its long-clip cells record the out-of-memory failure of every tested variant (BF16 offload, FP8 with and without the lifetime patch, expandable-segments allocation, VAE CPU offload); the first DiT forward of a 345-frame clip exceeds 24 GB before any weight is resident.

## Market-price cost columns in the top-level README

The top-level README prices each run at the GPU's median on-demand price per GPU-hour across cloud providers, as aggregated by GetDeploying, recorded with sources and retrieval date in `gpu-price-basis.json`. Cost per clip is price × GPUs used × measured seconds / 3600; cost per second of video divides by the clip length (124 or 345 frames at 24 fps). These are market snapshots for comparing hardware, not our invoices. The per-run `usd_per_clip` values in the results JSON remain the historical whole-node charges of the machines actually rented, which differ from the market basis for single-GPU runs on multi-GPU nodes.
