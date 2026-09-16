"""Apply the exact single-GPU VDN temporary-lifetime optimization."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REL=Path('python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn.py')
ORIGINAL_SHA256='ac3b7aaff8b77bb722498ae903b8412511959a6606fe3e3863c424639040b89f'
PATCHED_SHA256='9661344060b5ddaea6958819e0d99bffaad2fe4166dd6c31a805ba8792e294b2'


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:
        raise ValueError(f'Expected exactly one source anchor, found {source.count(old)}')
    return source.replace(old,new,1)


def modify(source):
    source=replace_once(source,'import functools\nimport math','import functools\nimport math\nimport os')
    source=replace_once(source,'_BF16 = torch.bfloat16',
        '_LEAN_LINEAR_LIFETIME = os.environ.get("LEANVDN_LINEAR_LIFETIME") == "1"\n\n_BF16 = torch.bfloat16')
    source=replace_once(source,'        query_by_frame = features(q_raw, proj="q", frame_major=True)',
        '        if _LEAN_LINEAR_LIFETIME:\n            if torch.is_grad_enabled() or q_raw.device.type != "cuda":\n                raise RuntimeError("VDN lifetime experiment needs CUDA inference")\n            if get_tp_world_size() != 1 or (torch.distributed.is_initialized()\n                    and torch.distributed.get_world_size() != 1):\n                raise RuntimeError("VDN lifetime experiment is single-GPU only")\n            if torch.cuda.is_current_stream_capturing():\n                raise RuntimeError("VDN lifetime experiment requires eager execution")\n        else:\n            query_by_frame = features(q_raw, proj="q", frame_major=True)')
    source=replace_once(source,'        del prepared\n        alpha = self.alpha(frame_mean, heads=heads)',
        '        del prepared\n        if _LEAN_LINEAR_LIFETIME:\n            # Statistics own their outputs; these feature buffers are now dead.\n            del key, value, key_by_frame, value_by_frame, beta_by_frame\n        alpha = self.alpha(frame_mean, heads=heads)')
    source=replace_once(source,'        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))',
        '        if _LEAN_LINEAR_LIFETIME:\n            # Q is consumed only here; avoid retaining it through frame statistics.\n            query_by_frame = features(q_raw, proj="q", frame_major=True)\n        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))')
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
        raise RuntimeError(f'Pinned VDN source hash mismatch: {before_sha}')
    receipt={'source':str(path),'before_sha256':before_sha,'after_sha256':digest(after),
             'state':state,'installed':bool(args.apply or before_sha==PATCHED_SHA256)}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
