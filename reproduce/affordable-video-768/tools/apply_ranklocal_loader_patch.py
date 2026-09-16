"""Apply the rank-local loader used by the multi-GPU RTX 5090 recipes.

Under sequence parallelism every rank keeps a full transformer. The pinned loader
builds each rank's BF16 tensors on its GPU before quantizing, which does not fit on
a 32 GB card. This patch keeps the checkpoint on the CPU, quantizes one module at a
time on the GPU and lets the ranks load one after another behind a local file lock;
a helper module validates the layout (TP 1, Ulysses = world size, folded unquantized
text encoder, MXFP8 or FP8) and writes a receipt per rank. Enabled through
AFFORDABLE_H3_RANKLOCAL_STREAM=1; the runner supplies the lock path and the receipt
directory. Three files: the loader, the text-encoder loader and the new helper.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=Path('python/sglang/multimodal_gen/runtime/loader')
REL=BASE/'fsdp_load.py'
ENCODER_REL=BASE/'component_loaders/text_encoder_loader.py'
HELPER_REL=BASE/'affordable_ranklocal_guard.py'
ORIGINAL_SHA256='9f01ae268ebd7d0e3abc40b295e25a6d8b24818a8bd110cb1f1e15ab4298c828'
PATCHED_SHA256='310b143cbd78b8db6496fa81569821aa4f6dd03a912ffc1607462c44b811ce94'
ENCODER_ORIGINAL_SHA256='d38ceb91ad8171cd6fda4a14ae9ccc758852bcfdc5b6d79131b8f2161c6b7c58'
ENCODER_PATCHED_SHA256='3089d143758036f67513b083fc8e5575be72945674d05e2c92df8c4f63f52ea2'
HELPER_SHA256='b0a8d62ca3b89ee01d81f7827dd8d94b7f3e5338df173d2640601ee0fc7013d7'


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:
        raise ValueError(f'Expected exactly one source anchor, found {source.count(old)}')
    return source.replace(old,new,1)


def modify(source):
    source=replace_once(source,'def maybe_load_fsdp_model(',
        'from sglang.multimodal_gen.runtime.loader.affordable_ranklocal_guard import guarded_serial_load\n\n\n@guarded_serial_load\ndef maybe_load_fsdp_model(')
    source=replace_once(source,'from collections import Counter, defaultdict','import os\nfrom collections import Counter, defaultdict')
    anchor='    weight_load_plan = weight_load_plan or WeightLoadPlan(checkpoint_load_device=device)'
    source=replace_once(source,anchor,anchor+'''
    affordable_stream = (
        os.environ.get("AFFORDABLE_H3_RANKLOCAL_STREAM") == "1"
        and type(model).__name__ == "MiniMaxH3DiTModel"
    )
    if affordable_stream:
        # Decorator has validated TP1, SP-only, CPU residency and quantization.
        # Each rank retains a full DiT; this is not weight sharding.
        weight_load_plan = WeightLoadPlan(checkpoint_load_device=torch.device("cpu"))
        logger.info("Experimental rank-local H3: CPU checkpoint, serial rank load, native per-module quantization")
''')
    source=replace_once(source,'    process_model_weights_after_loading(model)\n    model.post_load_weights()',
        '    process_model_weights_after_loading(model, process_device=device if affordable_stream else None)\n    model.post_load_weights()')
    ast.parse(source)
    return source


def modify_encoder(source):
    anchor='        encoder_dtype = self.component_load_precision(server_args, component_name)'
    source=replace_once(source,anchor,'''        from sglang.multimodal_gen.runtime.loader.affordable_ranklocal_guard import validate_encoder_layout
        validate_encoder_layout(encoder_config)
'''+anchor)
    ast.parse(source)
    return source


def helper_text():
    text=(HERE/'ranklocal_guard.py').read_text()
    if digest(text)!=HELPER_SHA256:
        raise RuntimeError('Shipped rank-local helper does not match the measured hash')
    return text


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    root=args.source.expanduser().resolve()
    plan=[(root/REL,ORIGINAL_SHA256,PATCHED_SHA256,modify),
          (root/ENCODER_REL,ENCODER_ORIGINAL_SHA256,ENCODER_PATCHED_SHA256,modify_encoder)]
    files=[];states=set()
    for path,original,patched,fn in plan:
        before=path.read_text();before_sha=digest(before)
        if before_sha==patched:
            after=before;state='already_applied'
        elif before_sha==original:
            after=fn(before);state='applied' if args.apply else 'verified'
            if digest(after)!=patched:
                raise RuntimeError(f'Generated patch hash does not match the measured source for {path.name}')
            if args.apply:path.write_text(after)
        else:
            raise RuntimeError(f'Pinned source hash mismatch for {path}: {before_sha}')
        states.add(state);files.append({'source':str(path),'before_sha256':before_sha,'after_sha256':digest(after),'state':state})
    helper=root/HELPER_REL;text=helper_text()
    if helper.exists():
        if digest(helper.read_text())!=HELPER_SHA256:
            raise RuntimeError(f'Unexpected file at {helper}; remove it or reset the checkout')
        helper_state='already_applied'
    else:
        helper_state='applied' if args.apply else 'verified'
        if args.apply:helper.write_text(text)
    files.append({'source':str(helper),'before_sha256':None,'after_sha256':HELPER_SHA256,'state':helper_state})
    state='already_applied' if states=={'already_applied'} and helper_state=='already_applied' else ('applied' if args.apply else 'verified')
    receipt={'source':str(root/REL),'files':files,'before_sha256':files[0]['before_sha256'],'after_sha256':files[0]['after_sha256'],
             'state':state,'installed':bool(args.apply or state=='already_applied')}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
