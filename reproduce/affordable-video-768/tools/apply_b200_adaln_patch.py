"""Verify and optionally apply the recorded B200 AdaLN patch to pinned SGLang."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

REL=Path('python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py')
ORIGINAL_SHA256='2a61d5c8b0418eed72fd3443cc47b6a06ed8512f6c834b608aa4e3e55e7489fa'
PATCHED_SHA256='df1d9caa615ee5f563ac2bd29e5ba6c433cfaaf1cca01b2c0ea1d693b5764f09'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(source,old,new):
    count=source.count(old)
    if count!=1:
        raise ValueError(f'Expected exactly one patch anchor, found {count}')
    return source.replace(old,new)


def modify(source):
    source=replace_once(source,'''        cache = self.adaln_cache
        if cache is None:
            return None
        # Keying costs one D2H sync per plan; compute the keys once and share''','''        if os.environ.get("LEANVDN_SGLANG_ADALN_CACHE") == "1" and self.adaln_cache is None:
            from sglang_adaln_extension import activate
            activate(self, step_timesteps)
        cache = self.adaln_cache
        if cache is None:
            return None
        # Keying costs one D2H sync per plan; compute the keys once and share''')
    source=replace_once(source,'''    def validate_lora_layers(self, layer_names: list[str]) -> None:
        if self._adaln_precomputed:''','''    def validate_lora_layers(self, layer_names: list[str]) -> None:
        if getattr(self.adaln_cache, "_leanvdn_runtime_cache", False):
            raise ValueError("Restart after changing LoRA with the LeanVDN loaded-projection cache")
        if self._adaln_precomputed:''')
    source=replace_once(source,'''        cache = self.adaln_cache
        if cache is None:
            return
        if cache.weight_files is None:''','''        cache = self.adaln_cache
        if getattr(cache, "_leanvdn_runtime_cache", False):
            raise ValueError("Restart after changing weights with the LeanVDN loaded-projection cache")
        if cache is None:
            return
        if cache.weight_files is None:''')
    ast.parse(source)
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='SGLang checkout root')
    parser.add_argument('--apply',action='store_true',help='Modify the checked source after verification')
    parser.add_argument('--receipt',type=Path,help='Optional JSON receipt path')
    args=parser.parse_args()
    source_root=args.source.expanduser().resolve()
    path=source_root/REL
    before=path.read_text()
    before_sha=digest(before)
    status='already_patched'
    patch_text=''
    if before_sha==ORIGINAL_SHA256:
        after=modify(before)
        after_sha=digest(after)
        if after_sha!=PATCHED_SHA256:
            raise RuntimeError(f'Patch output hash mismatch: expected {PATCHED_SHA256}, got {after_sha}')
        patch_text=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+str(REL),tofile='b/'+str(REL)))
        status='ready_to_apply'
        if args.apply:
            path.write_text(after)
            status='applied'
    elif before_sha!=PATCHED_SHA256:
        raise ValueError(f'Source differs from the pinned original and patched files: {before_sha}')
    receipt={'source_root':str(source_root),'path':str(REL),'before_sha256':before_sha,
             'after_sha256':PATCHED_SHA256,'status':status,'applied':status=='applied'}
    if args.receipt:
        receipt_path=args.receipt.expanduser().resolve()
        receipt_path.parent.mkdir(parents=True,exist_ok=True)
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        if patch_text:
            receipt_path.with_suffix('.patch').write_text(patch_text)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
