"""Run the VDN linear-branch readout on the current CUDA stream under sequence parallelism.

Under Ulysses the pinned source issues the linear readout and its all-to-all on a
private high-priority side stream. Repeated fixed-seed multi-GPU runs then produce
different decoded outputs; with the readout on the current stream the outputs
repeat byte for byte in a two-GPU diagnostic at unchanged request time. The change is
opt-in through LEANVDN_SP_MAIN_STREAM=1 so a patched checkout can still reproduce
the upstream behaviour.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REL=Path('python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn_attention.py')
ORIGINAL_SHA256='ff71881cec73e3656a4ed412a10847119bbd85d591cc0fee0f5f6a59096128b8'
PATCHED_SHA256='9753309a3808dbf5081360787facb71039154ad1ee0d687fc97a41f7f2a35ae2'


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:
        raise ValueError(f'Expected exactly one source anchor, found {source.count(old)}')
    return source.replace(old,new,1)


def modify(source):
    source=replace_once(source,'import functools\nimport logging\n','import functools\nimport logging\nimport os\n')
    source=replace_once(source,'_LINEAR_STREAMS: dict[int, torch.cuda.Stream] = {}\n',
        '_LINEAR_STREAMS: dict[int, torch.cuda.Stream] = {}\n_LEAN_SP_MAIN_STREAM = os.environ.get("LEANVDN_SP_MAIN_STREAM") == "1"\n')
    source=replace_once(source,'def _linear_branch_stream(device: torch.device) -> torch.cuda.Stream:\n    index = device.index',
        'def _linear_branch_stream(device: torch.device) -> torch.cuda.Stream:\n'
        '    if _LEAN_SP_MAIN_STREAM:\n'
        '        # LeanVDN: the side stream made repeated fixed-seed SP runs nondeterministic;\n'
        '        # the existing wait_stream calls become no-ops and the readout stays ordered.\n'
        '        return torch.cuda.current_stream(device)\n'
        '    index = device.index')
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
            raise RuntimeError('Generated patch hash does not match the recorded source')
        if args.apply:path.write_text(after)
    else:
        raise RuntimeError(f'Pinned VDN attention source hash mismatch: {before_sha}')
    receipt={'source':str(path),'before_sha256':before_sha,'after_sha256':digest(after),
             'state':state,'installed':bool(args.apply or before_sha==PATCHED_SHA256)}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
