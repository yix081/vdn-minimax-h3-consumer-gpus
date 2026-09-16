#!/usr/bin/env python3
"""Download the pinned model snapshots (about 145 GB) into ./models and verify every file.

    python3 download_models.py                # into ./models
    python3 download_models.py --verify-only  # recheck an existing cache, no network

The weights are MiniMax H3 derivatives under the MiniMax H3 Community License
(licenses/MiniMax-H3-Community-License-Agreement.txt); read it first. This wraps
reproduce/affordable-video-768/tools/stage_model.py, which pins the revisions
that every number in the README was measured with and writes the receipt
generate.py checks before loading.
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / 'reproduce/affordable-video-768/tools/stage_model.py'
NEEDED_GB = 160


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', type=Path, default=Path(os.environ.get('H3_MODELS', ROOT / 'models')))
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--use-auth-token', action='store_true', help='use your Hugging Face login if the repositories require it')
    p.add_argument('--python', type=Path, default=ROOT / '.venv-h3/bin/python')
    a = p.parse_args()
    if not a.python.is_file():
        sys.exit(f'{a.python} is missing; run ./install.sh first')
    a.models.mkdir(parents=True, exist_ok=True)
    if not a.verify_only:
        free_gb = shutil.disk_usage(a.models).free / 1e9
        if free_gb < NEEDED_GB:
            sys.exit(f'{a.models} has {free_gb:.0f} GB free; the snapshots need about {NEEDED_GB} GB')
    cmd = [str(a.python), str(STAGE), '--hf-home', str(a.models.resolve())]
    if a.verify_only:
        cmd.append('--verify-only')
    if a.use_auth_token:
        cmd.append('--use-auth-token')
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f'\nmodels verified in {a.models.resolve()}; generate.py finds them there by default')


if __name__ == '__main__':
    main()
