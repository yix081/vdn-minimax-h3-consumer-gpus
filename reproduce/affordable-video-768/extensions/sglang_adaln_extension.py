"""Exact, schedule-keyed AdaLN caching for the pinned SGLang FP8 VDN runtime.

The original loaded projection modules are the numerical oracle. Keep them on
CPU between schedule builds, and run the same quantization method/shape when a
new schedule is requested. This experimental integration supports pure Ulysses SP with TP=1; every rank keeps its own exact table.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group, get_world_group
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import (
    MiniMaxH3AdalnCache,
    _plan_key,
)


def _bytes(module):
    return sum(t.numel() * t.element_size() for t in module.parameters()) + sum(
        t.numel() * t.element_size() for t in module.buffers()
    )


def _memory():
    return {"allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved()}


def _write_report(data):
    directory = os.environ.get("LEANVDN_ADALN_REPORT_DIR")
    if directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"cache-{os.getpid()}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(path)


class LoadedProjectionCache(MiniMaxH3AdalnCache):
    """Reuse upstream slot management, lookups and fixed-shape cache storage."""

    _leanvdn_runtime_cache = True

    def __init__(self, model, steps):
        sp_size = get_sp_group().world_size
        world = get_world_group()
        if get_tp_world_size() != 1 or sp_size not in (1, 2, 4, 8) or world.world_size != sp_size:
            raise ValueError("Experimental cache supports pure Ulysses SP1/2/4/8 and TP1 only")
        if is_layerwise_offloaded_module(model):
            raise ValueError("LeanVDN AdaLN cache cannot replace offload-managed DiT modules")
        if model.arch.adaln_curve_grid is not None:
            raise ValueError("Curve AdaLN checkpoints are outside this exact-cache integration")
        super().__init__(model.arch, weight_files=[],
                         max_plans=envs.SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS,
                         max_plan_width=4, host_cache_bytes=0, precision="match")
        # An ordinary list intentionally keeps CPU source modules outside the
        # registered GPU cache tree. They are retained for safe schedule rebuilds.
        self.sources = [block.adaln_proj for block in model.blocks]
        self.sources.append(model.final_layer.adaln_proj)
        if any(source is None for source in self.sources):
            raise ValueError("AdaLN projections have already been replaced")
        self.reference = None
        self.report = {"method": "loaded_fp8_projection_cache", "version": 2, "sp_size": sp_size, "rank": world.rank,
                       "precision": "native loaded projection, unchanged",
                       "source_bytes": sum(_bytes(m) for m in self.sources),
                       "projection_methods": sorted({type(m.linear.quant_method).__name__
                                                     for m in self.sources}),
                       "before": _memory(), "builds": [],
                       "invalidation": "schedule changes rebuild; weight/LoRA updates require restart",
                       "first_build_in_warmup": True}
        if os.environ.get("LEANVDN_ADALN_VERIFY") == "1":
            embeddings = [torch.nn.functional.silu(model.time_embedder(t)).bfloat16()
                          for t in steps]
            self.reference = [{_plan_key(t): m.project_local(x).detach().cpu()
                               for t, x in zip(steps, embeddings)} for m in self.sources]
        # Transfer the loaded FP8 representations, including their scales,
        # without decoding/requantizing the checkpoint.
        for source in self.sources:
            source.cpu()
        for block in model.blocks:
            block.adaln_proj = None
        model.final_layer.adaln_proj = None
        self.load(model.video_patch_proj.weight.device)
        torch.cuda.empty_cache()
        self.report.update(after_enable=_memory(), table_bytes=sum(
            t.numel() * t.element_size() for t in self.buffers()))
        _write_report(self.report)

    @torch.no_grad()
    def _project_plans_from_checkpoint(self, slots, *, tp_size, tp_rank, device):
        if tp_size != 1 or tp_rank != 0:
            raise ValueError("Tensor-parallel projection reconstruction is not supported")
        begin = time.perf_counter()
        checks = []
        for layer, source in enumerate(self.sources):
            # Keep original CPU storage alive so restoration needs no D2H copy.
            cpu_params = [(p, p.data) for p in source.parameters()]
            cpu_buffers = [(m, name, b) for m in source.modules()
                           for name, b in m._buffers.items() if b is not None]
            source.to(device)
            try:
                for ordinal, (slot, length, embedding) in enumerate(slots):
                    projected = source.project_local(embedding)
                    destination = (self.block_params[slot, :length, layer]
                                   if layer < self.num_layers
                                   else self.final_params[slot, :length])
                    destination.copy_(projected)
                    if self.reference is not None:
                        key = _plan_key(self.plan_timesteps[slot, :length])
                        reference = self.reference[layer][key].to(device)
                        equal = torch.equal(reference, destination)
                        checks.append({"layer": layer, "plan": ordinal, "equal": equal,
                                       "max_abs_error": float((reference.float() - destination.float()).abs().max())})
                        if not equal:
                            raise RuntimeError(f"AdaLN parity failed at layer {layer}, plan {ordinal}")
            finally:
                for parameter, cpu in cpu_params:
                    parameter.data = cpu
                for module, name, cpu in cpu_buffers:
                    module._buffers[name] = cpu
        torch.cuda.synchronize(device)
        self.report["builds"].append({"seconds": time.perf_counter() - begin,
                                     "plans": len(slots), "checks": checks,
                                     "all_equal": all(c["equal"] for c in checks) if checks else None,
                                     "memory": _memory()})
        self.reference = None
        _write_report(self.report)


@torch.no_grad()
def activate(model, step_timesteps):
    # Keep synthetic short warmup native; verify all eight plans on first real warm request.
    if len(step_timesteps) != 8:
        return
    if model.adaln_cache is not None:
        raise ValueError("Do not combine the LeanVDN cache with another AdaLN cache")
    model.adaln_cache = LoadedProjectionCache(model, step_timesteps)
    model._adaln_precomputed = True
