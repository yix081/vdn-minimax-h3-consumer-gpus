#!/usr/bin/env python3
"""Generate one VDN-MiniMax-H3 clip on the GPU in this machine.

    python3 generate.py --prompt "A close-up of a hand pouring water into a glass."

The script detects the GPU, picks the recipe we measured for it, puts the pinned
SGLang checkout into exactly the patch state that recipe was measured with, and
runs one request through the benchmark runner in reproduce/affordable-video-768.
It needs ./install.sh and ./download_models.py to have run first. Standard
library only; the heavy work happens in the .venv-h3 interpreter it launches.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPRO = ROOT / 'reproduce/affordable-video-768'
SOURCE_REVISION = 'd72e59508b7554045cb51827f9b8d0f08c7a3abc'
FRAMES = {'short': 124, 'long': 345}

# GPU key -> (substring of the nvidia-smi name, recipe per clip length or per GPU count)
GPUS = {
    'rtx4090':     ('RTX 4090',      {'short': 'rtx4090-short.json', 'long': None}),
    'rtx5090':     ('RTX 5090',      {'short': 'rtx5090-short.json', 'long': 'rtx5090-long.json'}),
    'rtx-pro5000': ('RTX PRO 5000',  {'short': 'rtx-pro5000-short.json', 'long': 'rtx-pro5000-long.json'}),
    'l40s':        ('L40S',          {'short': 'l40s-short.json', 'long': 'l40s-long.json'}),
    'rtx-pro6000': ('RTX PRO 6000',  {'short': 'rtx-pro6000-short.json', 'long': 'rtx-pro6000-long.json'}),
    'h100':        ('H100',          {n: f'h100-{n}.json' for n in (1, 2, 4, 8)}),
    'b200':        ('B200',          {n: f'b200-{n}-native.json' for n in (1, 2, 4, 8)}),
}
DETECT_ORDER = ['rtx-pro5000', 'rtx-pro6000', 'rtx5090', 'rtx4090', 'l40s', 'h100', 'b200']

# patch id used in recipes -> tool that applies it; every tool is idempotent and hash-checked
PATCH_TOOLS = {
    'streamed-quant-loader': 'apply_stream_quant_patch.py',
    'linear-lifetime': 'apply_linear_lifetime_patch.py',
    'sp-main-stream': 'apply_sp_main_stream_patch.py',
    'exact-loaded-projection-adaln-cache': 'apply_b200_adaln_patch.py',
    'ranklocal-loader': 'apply_ranklocal_loader_patch.py',
}
# the only files any patch touches; reset to the pinned revision before applying a recipe's patches
PATCHED_FILES = [
    'python/sglang/multimodal_gen/runtime/loader/fsdp_load.py',
    'python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn.py',
    'python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn_attention.py',
    'python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py',
    'python/sglang/multimodal_gen/runtime/loader/component_loaders/text_encoder_loader.py',
]
# files a patch adds; removed before a recipe's patches are applied
ADDED_FILES = ['python/sglang/multimodal_gen/runtime/loader/affordable_ranklocal_guard.py']


def fail(msg):
    print(f'error: {msg}', file=sys.stderr)
    sys.exit(2)


def detect_gpu():
    try:
        names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).split('\n')
    except (OSError, subprocess.CalledProcessError):
        fail('nvidia-smi not found or failed; pass --gpu explicitly')
    names = [n.strip() for n in names if n.strip()]
    for key in DETECT_ORDER:
        if any(GPUS[key][0] in n for n in names):
            return key, names
    fail(f'no supported GPU found in {names}; supported: {", ".join(GPUS)}')


def pick_recipe(gpu, length, gpus):
    match, table = GPUS[gpu]
    if gpu in ('h100', 'b200'):
        if gpus not in table:
            fail(f'{gpu} recipes exist for 1, 2, 4 or 8 GPUs, not {gpus}')
        return REPRO / 'recipes' / table[gpus]
    if gpus != 1:
        candidate = REPRO / 'recipes' / f'{gpu}-{gpus}-{length}.json'
        if candidate.is_file():
            return candidate
        fail(f'{gpu} has no released recipe for {gpus} GPUs and the {length} clip (see README)')
    name = table[length]
    if name is None:
        fail(f'{gpu}: the {length} clip ({FRAMES[length]} frames) does not fit in this card\'s memory in any tested configuration; use --length short')
    return REPRO / 'recipes' / name


def structured_prompt(a):
    if a.raw_prompt:
        return a.raw_prompt
    return (f'integrated_multimodal_description: {a.prompt.strip()}\n\n'
            f'overall_soundscape: {a.sound.strip()}\n\n'
            f'non_diegetic_music: {a.music.strip()}')


def set_patch_state(python, source, patches, work, dry):
    cmd = ['git', '-C', str(source), 'checkout', '--'] + PATCHED_FILES
    print('+', ' '.join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)
    for added in ADDED_FILES:
        path = Path(source) / added
        if path.exists():
            print('+ rm', path)
            if not dry:
                path.unlink()
    for patch in patches:
        cmd = [str(python), str(REPRO / 'tools' / PATCH_TOOLS[patch]), '--source', str(source), '--apply',
               '--receipt', str(work / f'patch-{patch}.json')]
        print('+', ' '.join(cmd))
        if not dry:
            subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prompt', help='what happens in the clip (one or two sentences)')
    p.add_argument('--sound', default='Natural ambient sound matching the visible scene. No speech.', help='soundscape line of the prompt')
    p.add_argument('--music', default='N/A', help='non-diegetic music line of the prompt')
    p.add_argument('--raw-prompt', help='full structured VDN prompt; overrides --prompt/--sound/--music')
    p.add_argument('--length', choices=list(FRAMES), default='short', help='short = 124 frames (5.2 s), long = 345 frames (14.4 s)')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--gpu', choices=list(GPUS), help='default: detect with nvidia-smi')
    p.add_argument('--gpus', type=int, default=1, help='GPU count (1, 2, 4, 8) on H100, B200 and RTX 5090')
    p.add_argument('--formal', action='store_true', help="run the recipe's measured protocol (warm-ups, then three timed requests) instead of one request; prints all timings and whether the decoded outputs repeated")
    p.add_argument('--out', type=Path, help='where to put the MP4 (default outputs/<timestamp>.mp4)')
    p.add_argument('--work', type=Path, help='run directory for the runner\'s report and receipts (default runs/<timestamp>)')
    p.add_argument('--models', type=Path, default=Path(os.environ.get('H3_MODELS', ROOT / 'models')), help='model cache from download_models.py')
    p.add_argument('--sglang-source', type=Path, default=Path(os.environ.get('SGLANG_SOURCE', ROOT / 'sglang-d72e595')))
    p.add_argument('--python', type=Path, default=ROOT / '.venv-h3/bin/python', help='interpreter created by install.sh')
    p.add_argument('--dry-run', action='store_true', help='print the plan and the exact commands, change nothing')
    a = p.parse_args()
    if not a.prompt and not a.raw_prompt:
        fail('--prompt (or --raw-prompt) is required')

    gpu = a.gpu
    names = None
    if gpu is None:
        gpu, names = detect_gpu()
    recipe_path = pick_recipe(gpu, a.length, a.gpus)
    recipe = json.loads(recipe_path.read_text())
    frames = FRAMES[a.length]

    stamp = time.strftime('%Y%m%d-%H%M%S')
    work = (a.work or ROOT / 'runs' / stamp).resolve()
    out = (a.out or ROOT / 'outputs' / f'{stamp}-{gpu}-{a.length}.mp4').resolve()
    if work.exists():
        fail(f'{work} exists; the runner refuses to overwrite a run directory')

    # sanity checks that give a clear message before the venv interpreter starts
    if not a.dry_run:
        if not a.python.is_file():
            fail(f'{a.python} is missing; run ./install.sh first')
        if not (a.models / 'staging-receipt.json').is_file():
            fail(f'{a.models} has no staging receipt; run python3 download_models.py first')
        try:
            head = subprocess.check_output(['git', '-C', str(a.sglang_source), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            fail(f'{a.sglang_source} is not the pinned SGLang checkout; run ./install.sh')
        if head != SOURCE_REVISION:
            fail(f'{a.sglang_source} is at {head[:12]}, expected {SOURCE_REVISION[:12]}')
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        nvcc = Path(cuda_home, 'bin/nvcc') if cuda_home else (Path(shutil.which('nvcc')) if shutil.which('nvcc') else Path('/usr/local/cuda/bin/nvcc'))
        if not nvcc.is_file():
            fail('no CUDA 13 toolkit found (nvcc via CUDA_HOME, PATH or /usr/local/cuda); SGLang compiles its kernels at first use')

    # one request, no warmups; the recipe is otherwise byte-for-byte the measured one
    run_recipe = dict(recipe)
    if a.formal:
        run_recipe['performance_protocol'] = recipe.get('performance_protocol', {'feasibility': 1, 'warmups': 2, 'timings': 3})
    else:
        run_recipe['performance_protocol'] = {'feasibility': 0, 'warmups': 0, 'timings': 1}
    case = [{'id': 'request', 'prompt': structured_prompt(a), 'seed': a.seed,
             'duration_seconds': frames / 24, 'expected_frames': frames}]
    stage = work.parent / f'{work.name}.input'

    env = dict(os.environ)
    env.update(HF_HOME=str(a.models.resolve()), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # SGLang builds its JIT kernels by calling `ninja` (a package in the frozen set) as a
    # command, so the interpreter's bin directory has to be on PATH, as if the venv were active.
    # (no resolve(): .venv-h3/bin/python is a symlink to the system interpreter)
    env['PATH'] = os.pathsep.join([os.path.dirname(os.path.abspath(a.python)), env.get('PATH', '')])
    if 'CUDA_VISIBLE_DEVICES' not in env:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(recipe['server_args']['num_gpus']))
    cmd = [str(a.python), str(REPRO / 'generate.py'), '--output', str(work), '--recipe', str(stage / 'recipe.json'),
           '--cases', str(stage / 'case.json'), '--suite', 'performance']

    print(json.dumps({'gpu': gpu, 'detected': names, 'recipe': recipe_path.name, 'patches': recipe.get('patches', []),
                      'environment': recipe.get('environment', {}), 'frames': frames, 'seed': a.seed,
                      'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'], 'output': str(out), 'run_dir': str(work)}, indent=2))
    if not a.dry_run:
        stage.mkdir(parents=True, exist_ok=True)
        (stage / 'recipe.json').write_text(json.dumps(run_recipe, indent=2) + '\n')
        (stage / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
        work.parent.mkdir(parents=True, exist_ok=True)
    set_patch_state(a.python, a.sglang_source, recipe.get('patches', []), stage, a.dry_run)
    print('+', ' '.join(cmd))
    if a.dry_run:
        return
    started = time.perf_counter()
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    report = json.loads((work / 'report.json').read_text())
    if report.get('status') != 'completed' or not report.get('runs'):
        fail(f'run did not complete; see {work}/report.json')
    timings = [r for r in report['runs'] if r['kind'] == 'timing']
    row = timings[-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(row['output_file_path'], out)
    summary = {'output': str(out), 'request_seconds': round(row['complete_request_seconds'], 1),
               'load_seconds': round(report.get('load_seconds', 0), 1), 'total_seconds': round(time.perf_counter() - started, 1),
               'gpu_peak_mb': row.get('peak_memory_mb'), 'sha256': row['media']['sha256']}
    if a.formal:
        seconds = [r['complete_request_seconds'] for r in timings]
        summary.update(timing_seconds=[round(v, 2) for v in seconds], mean_seconds=round(sum(seconds) / len(seconds), 2),
                       warmup_seconds=[round(r['complete_request_seconds'], 2) for r in report['runs'] if r['kind'] == 'warmup'],
                       decoded_outputs_identical=len({(r['decoded_video_sha256'], r['decoded_audio_sha256']) for r in timings}) == 1)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
