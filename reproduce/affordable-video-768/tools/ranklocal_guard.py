"""Experimental H3 loader guard; stdlib importable for CPU-only fixtures.

This is a placement extension, not a host-memory admission controller. The
launcher must still own cgroup telemetry, a finite timeout, and a host budget.
"""
import contextlib
import fcntl
import functools
import inspect
import json
import os
from pathlib import Path
import time

ENABLE = "AFFORDABLE_H3_RANKLOCAL_STREAM"


def validate_layout(s):
    """Reject unreviewed layouts before constructing the DiT or taking a lock."""
    w = s["world"]
    required = {
        "initialized": True, "nnodes": 1, "tp": 1, "sp": w,
        "ulysses": w, "ring": 1, "kv_gather": 1, "dp": 1, "cfg": 1,
        "pp": 1, "num_gpus": w, "fsdp": False, "starts_cpu": True,
        "local_safetensors": True, "device_type": "cuda", "encoder_policy": "fold",
        "disaggregated": False, "lora": False,
        "filtered_checkpoint": False, "adaln": False,
    }
    if w not in (2, 4, 8):
        raise ValueError("Rank-local experiment admits only 2/4/8 ranks")
    for name, value in required.items():
        if s.get(name) != value:
            raise ValueError(f"Unsupported {name}: {s.get(name)!r}; expected {value!r}")
    if not (0 <= s["rank"] < w and 0 <= s["local_rank"] < w):
        raise ValueError("Invalid rank")
    if (s["visible_devices"] < w or s["device_index"] != s["current_device"]
            or s["device_index"] != s["assigned_device"]):
        raise ValueError("Each worker must load on its own current CUDA device")
    if (s["capability"], s["quantization"]) not in ((89, "fp8"), (120, "mxfp8")):
        raise ValueError("Only reviewed SM89/FP8 and SM120/MXFP8 combinations admitted")


@contextlib.contextmanager
def serial_load(path, timeout):
    """One shared local lock, held over the entire independent DiT load.

    No barriers inside this lock. Never delete/recreate the lock file while
    workers exist: that could let ranks lock different inodes.
    """
    path = Path(path)
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("Use one absolute lock path in an existing shared job directory")
    local_filesystem(path.parent)
    if not 0 < timeout <= 3600:
        raise ValueError("Lock timeout must be finite and at most 3600 seconds")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for serial DiT load")
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def local_filesystem(path, mountinfo=Path('/proc/self/mountinfo')):
    target=str(Path(path).resolve());matches=[]
    for line in mountinfo.read_text().splitlines():
        left,right=line.split(' - ',1);mount=left.split()[4].replace('\\040',' ')
        if target==mount or target.startswith(mount.rstrip('/')+'/'):
            matches.append((len(mount),right.split()[0]))
    if not matches:raise ValueError('Cannot establish lock filesystem')
    fs=max(matches)[1]
    if fs not in ('ext2','ext3','ext4','xfs','btrfs','overlay','tmpfs','zfs'):
        raise ValueError(f'Lock must use reviewed local filesystem, got {fs}')
    return fs


def receipt(name,row):
    directory=Path(os.environ['AFFORDABLE_RANK_RECEIPTS'])
    if not directory.is_absolute():raise ValueError('Rank receipt directory must be absolute')
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/name;temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(target)


def runtime_snapshot(bound):
    import torch
    from sglang.multimodal_gen.runtime.distributed import parallel_state as p
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args
    a = get_global_server_args()
    device = bound["device"]
    quant = bound["init_params"].get("quant_config")
    capability = torch.cuda.get_device_capability(device)
    return dict(
        initialized=torch.distributed.is_initialized(),
        world=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(),
        local_rank=p.get_world_group().local_rank, nnodes=a.nnodes,
        num_gpus=a.num_gpus, tp=p.get_tp_world_size(), sp=p.get_sp_world_size(),
        ulysses=p.get_ulysses_parallel_world_size(), ring=p.get_ring_parallel_world_size(),
        kv_gather=a.kv_gather_degree or 1, dp=p.get_dp_world_size(),
        cfg=p.get_classifier_free_guidance_world_size(), pp=p.get_pipeline_parallel_world_size(),
        fsdp=bound["fsdp_inference"], starts_cpu=bound["component_starts_on_cpu"],
        local_safetensors=(bound["weights_iterator"] is None
            and bool(bound["weight_dir_list"])
            and all(Path(name).is_file() and Path(name).suffix == ".safetensors"
                    for name in bound["weight_dir_list"])),
        device_type=device.type, device_index=device.index,
        assigned_device=p.get_world_group().device.index,
        current_device=torch.cuda.current_device(), visible_devices=torch.cuda.device_count(),
        encoder_policy=a.encoder_parallel, disaggregated=bool(a.disagg_mode),
        lora=bool(a.lora_path), capability=capability[0] * 10 + capability[1],
        quantization=quant.get_name() if quant is not None else None,
        filtered_checkpoint=bound['checkpoint_key_filter'] is not None,
        adaln=(any(k.startswith('adaln') and v is not None for k,v in bound['init_params'].items())
               or getattr(getattr(bound['init_params'].get('config'),'arch_config',None),'adaln_curve_grid',None) is not None
               or bool(getattr(a,'minimax_h3_adaln_online',False))
               or getattr(a,'minimax_h3_adaln_cache_path',None) is not None
               or os.environ.get('LEANVDN_SGLANG_ADALN_CACHE','0')!='0'),
        dist_timeout=a.dist_timeout,
    )


def guarded_serial_load(fn):
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if os.environ.get(ENABLE) != "1":
            return fn(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        b = bound.arguments
        if b["model_cls"].__name__ != "MiniMaxH3DiTModel":
            return fn(*args, **kwargs)
        if os.environ.get("LEANVDN_STREAM_QUANT_LOAD") == "1":
            raise ValueError("Do not combine old and new stream loader switches")
        snapshot=runtime_snapshot(b);validate_layout(snapshot)
        lock = os.environ["AFFORDABLE_H3_LOAD_LOCK"]
        timeout = float(os.environ["AFFORDABLE_H3_LOAD_LOCK_TIMEOUT_S"])
        if not timeout < snapshot['dist_timeout']:
            raise ValueError('Lock timeout must be shorter than distributed timeout')
        receipt(f"rank-{snapshot['rank']}-loader.json",snapshot|{'at':time.time(),'lock_timeout':timeout,'lock_filesystem':local_filesystem(Path(lock).parent)})
        started=time.monotonic()
        with serial_load(lock, timeout):
            acquired=time.monotonic();model=fn(*args, **kwargs)
            receipt(f"rank-{snapshot['rank']}-loaded.json",snapshot|{'at':time.time(),'lock_wait_seconds':acquired-started,'rank_load_seconds':time.monotonic()-acquired})
            return model
    return wrapped


def validate_encoder_layout(config):
    """Called after native folding resolution, before text model construction."""
    if os.environ.get(ENABLE) != "1":
        return
    from sglang.multimodal_gen.runtime.distributed import get_world_size
    from sglang.multimodal_gen.runtime.distributed import get_world_rank
    from sglang.multimodal_gen.runtime.models.encoders.base import get_folding_tp_group
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args
    if (get_global_server_args().encoder_parallel != "fold"
            or config.parallel_folding_mode not in ("world", "replica", "sp")
            or get_folding_tp_group(config).world_size != get_world_size()
            or config.quant_config is not None):
        raise ValueError("Rank-local H3 requires an actually folded, unquantized text encoder")
    receipt(f'rank-{get_world_rank()}-encoder.json',{'at':time.time(),'rank':get_world_rank(),'encoder_tp_group_size':get_folding_tp_group(config).world_size,'world':get_world_size(),'parallel_folding_mode':config.parallel_folding_mode,'quantization':None})
