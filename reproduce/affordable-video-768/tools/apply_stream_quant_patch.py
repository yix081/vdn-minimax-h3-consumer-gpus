"""Apply the exact streamed quantized-weight loader used by the 24/32/48 GB recipes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REL=Path('python/sglang/multimodal_gen/runtime/loader/fsdp_load.py')
ORIGINAL_SHA256='9f01ae268ebd7d0e3abc40b295e25a6d8b24818a8bd110cb1f1e15ab4298c828'
PATCHED_SHA256='5d726735b4eeb634c36a6d3d3bf19f47dd54f3ea00e3f46cf9ff721d8a676205'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:
        raise ValueError(f'Expected exactly one source anchor, found {source.count(old)}')
    return source.replace(old,new,1)


def modify(source):
    source=replace_once(source,'from collections import Counter, defaultdict',
                        'import os\nfrom collections import Counter, defaultdict')
    old='    weight_load_plan = weight_load_plan or WeightLoadPlan(checkpoint_load_device=device)'
    new=old+'\n    lean_streamed_quant = False\n    if os.environ.get("LEANVDN_STREAM_QUANT_LOAD") == "1" and type(model).__name__ == "MiniMaxH3DiTModel":\n        quant = init_params.get("quant_config")\n        quant_name = quant.get_name() if quant is not None else None\n        if quant_name not in ("fp8", "mxfp8", "kitchen_int8"):\n            raise ValueError("Streamed H3 quantization admits only FP8/MXFP8/kitchen_int8")\n        if use_fsdp or (torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1):\n            raise ValueError("Streamed H3 quantization is initially single-GPU only")\n        if device.type != "cuda" or weights_iterator is not None:\n            raise ValueError("Streamed H3 quantization needs local safetensors and CUDA")\n        weight_load_plan = WeightLoadPlan(checkpoint_load_device=torch.device("cpu"))\n        lean_streamed_quant = True\n        logger.info("LeanVDN streamed quantization: CPU checkpoint; native per-module GPU postprocess; final starts_on_cpu=%s", component_starts_on_cpu)\n'
    source=replace_once(source,old,new)
    source=replace_once(source,'    process_model_weights_after_loading(model)\n    model.post_load_weights()',
                        '    process_model_weights_after_loading(model, process_device=device if lean_streamed_quant else None)\n    model.post_load_weights()')
    source=replace_once(source,'    # 4. deferred cpu offload\n    if defer_cpu_placement:',
                        '    if lean_streamed_quant and not component_starts_on_cpu:\n        _move_to_device_preserving_meta(model, device)\n\n    # 4. deferred cpu offload\n    if defer_cpu_placement:')
    ast.parse(source)
    return source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    path=args.source.expanduser().resolve()/REL
    before=path.read_text(); before_sha=digest(before)
    if before_sha==PATCHED_SHA256:
        after=before; state='already_applied'
    elif before_sha==ORIGINAL_SHA256:
        after=modify(before); state='applied' if args.apply else 'verified'
        if digest(after)!=PATCHED_SHA256:
            raise RuntimeError('Generated patch hash does not match the measured source')
        if args.apply:path.write_text(after)
    else:
        raise RuntimeError(f'Pinned loader hash mismatch: {before_sha}')
    receipt={'source':str(path),'before_sha256':before_sha,'after_sha256':digest(after),
             'state':state,'installed':bool(args.apply or before_sha==PATCHED_SHA256)}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
