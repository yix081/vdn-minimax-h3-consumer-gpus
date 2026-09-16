#!/usr/bin/env bash
# Set up everything generate.py needs, next to this script:
#   .venv-h3/          Python 3.12 environment with the exact package versions we measured
#   sglang-d72e595/    SGLang at the pinned revision, installed editable into that environment
# The three loader/memory patches are applied per run by generate.py, so this checkout stays
# byte-identical to upstream until a recipe needs a patch. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
SRC="${SGLANG_SOURCE:-$PWD/sglang-d72e595}"
REV=d72e59508b7554045cb51827f9b8d0f08c7a3abc
REQ=reproduce/affordable-video-768/requirements-frozen.txt

for tool in git ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || { echo "missing: $tool (Debian/Ubuntu: sudo apt-get install -y git ffmpeg python3.12-venv python3.12-dev build-essential pkg-config libnuma1)"; exit 1; }
done
"$PY" -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version' 2>/dev/null \
  || { echo "need Python 3.12 (found: $("$PY" --version 2>&1)); set PYTHON=python3.12"; exit 1; }
command -v nvcc >/dev/null || [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] \
  || echo "warning: no nvcc found; generate.py needs the CUDA 13 toolkit because SGLang compiles its kernels at first use"

if [ ! -d "$SRC/.git" ]; then
  echo "fetching SGLang $REV into $SRC"
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/sgl-project/sglang.git
  git -C "$SRC" fetch -q --depth 1 origin "$REV"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$REV" ] || { echo "$SRC is not at $REV"; exit 1; }

[ -x .venv-h3/bin/python ] || "$PY" -m venv .venv-h3
.venv-h3/bin/python -m pip install -q --upgrade pip uv
echo "installing the frozen package set (CUDA 13 wheels, several GB)"
.venv-h3/bin/uv pip install -q --python .venv-h3/bin/python --no-deps -r "$REQ"
SGLANG_BUILD_RUST_EXTS=none .venv-h3/bin/uv pip install -q --python .venv-h3/bin/python --no-deps -e "$SRC/python"
.venv-h3/bin/python -c 'import sglang, torch; print("sglang", sglang.__file__); print("torch", torch.__version__, "cuda", torch.version.cuda)'

echo
echo "done. next:"
echo "  python3 download_models.py            # about 145 GB of pinned weights into ./models"
echo "  python3 generate.py --prompt '...'    # one clip on the GPU in this machine"
