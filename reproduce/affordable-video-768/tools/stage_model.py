"""Stage and verify the measured model revisions in a dedicated offline HF cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

MANIFEST = Path(__file__).resolve().parents[1] / 'model-manifest.json'


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('models'):
        raise ValueError('Invalid model manifest')
    for spec in manifest['models'].values():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', spec['repo']):
            raise ValueError('Invalid model repository')
        if not re.fullmatch(r'[a-f0-9]{40}', spec['revision']):
            raise ValueError('Model revision must be an immutable commit')
        seen = set()
        for record in spec['files']:
            name = record['file']
            parts = PurePosixPath(name)
            if (parts.is_absolute() or '..' in parts.parts or '\\' in name
                    or name != parts.as_posix() or name in seen):
                raise ValueError(f'Invalid or duplicate model path: {name}')
            if not re.fullmatch(r'[a-f0-9]{64}', record['sha256']):
                raise ValueError(f'Invalid file hash: {name}')
            if not isinstance(record['bytes'], int) or record['bytes'] < 0:
                raise ValueError(f'Invalid file size: {name}')
            seen.add(name)
    return manifest


def repo_cache(cache: Path, spec: dict) -> Path:
    return cache / ('models--' + spec['repo'].replace('/', '--'))


def verify_snapshot(snapshot: Path, spec: dict) -> None:
    for record in spec['files']:
        path = snapshot / record['file']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise RuntimeError(f'Missing or wrong-sized model file: {path}')
        if sha256(path) != record['sha256']:
            raise RuntimeError(f'Model checksum mismatch: {path}')


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


COMPLETE_SNAPSHOT_REPOS = {'OpenVDN/vdn-minimax-h3', 'kevin-mi/VDN-H3-overlay'}


def stage(hf_home: Path, *, manifest_path: Path = MANIFEST,
          verify_only: bool = False, use_auth_token: bool = False,
          downloader=None) -> dict:
    manifest = load_manifest(manifest_path)
    hf_home = hf_home.expanduser().resolve()
    cache = hf_home / 'hub'
    # Do not retarget an existing cache to another model version.
    for spec in manifest['models'].values():
        ref = repo_cache(cache, spec) / 'refs/main'
        if ref.exists() and ref.read_text().strip() != spec['revision']:
            raise RuntimeError(f'Cache contains another revision: {ref}; use a fresh --hf-home')
    if not verify_only and downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    verified = {}
    for name, spec in manifest['models'].items():
        snapshot = repo_cache(cache, spec) / 'snapshots' / spec['revision']
        if not verify_only:
            # SGLang resolves the base model with snapshot_download, and the hub's offline
            # mode refuses a snapshot that lacks any repository file, so that repository is
            # fetched whole (about 4 GiB beyond the manifest). MiniMax-H3 is only read file
            # by file for its text encoder, so only its manifest files are fetched: the
            # repository holds several variants, 464 GiB in total, of which the runtime
            # reads 62 GiB. Every manifest file is then verified by size and SHA-256.
            patterns = None if spec['repo'] in COMPLETE_SNAPSHOT_REPOS else [record['file'] for record in spec['files']]
            actual = Path(downloader(
                repo_id=spec['repo'], revision=spec['revision'],
                cache_dir=str(cache), max_workers=6,
                allow_patterns=patterns,
                token=True if use_auth_token else False,
            ))
            if actual.resolve() != snapshot.resolve():
                raise RuntimeError(f'Downloader returned an unexpected snapshot for {name}')
        verify_snapshot(snapshot, spec)
        verified[name] = {'repo': spec['repo'], 'revision': spec['revision'],
                          'files': len(spec['files']),
                          'bytes': sum(record['bytes'] for record in spec['files'])}
    # Only publish offline refs after every repository passes verification.
    for spec in manifest['models'].values():
        ref = repo_cache(cache, spec) / 'refs/main'
        if verify_only:
            if not ref.is_file() or ref.read_text().strip() != spec['revision']:
                raise RuntimeError(f'Offline ref is not pinned: {ref}')
        else:
            ref.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves a different writer's existing ref.
            try:
                with ref.open('x') as stream:
                    stream.write(spec['revision'])
            except FileExistsError:
                if ref.read_text().strip() != spec['revision']:
                    raise RuntimeError(f'Offline ref changed during staging: {ref}')
    receipt = {'status': 'verified', 'manifest_sha256': sha256(manifest_path),
               'hf_home': str(hf_home), 'models': verified,
               'runtime_environment': {'HF_HOME': str(hf_home),
                                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}}
    if not verify_only:
        write_json(hf_home / 'staging-receipt.json', receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-home', type=Path, required=True,
                        help='Dedicated cache directory; exported as HF_HOME when generating')
    parser.add_argument('--verify-only', action='store_true',
                        help='Hash existing files and check refs without network access or writes')
    parser.add_argument('--use-auth-token', action='store_true',
                        help='Use an existing Hugging Face login for repositories requiring access')
    args = parser.parse_args()
    receipt = stage(args.hf_home, verify_only=args.verify_only,
                    use_auth_token=args.use_auth_token)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
