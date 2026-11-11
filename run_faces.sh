#!/usr/bin/env bash
# ============================================================
#  AI_FaceRecognitor - Runtime launcher (Linux mirror of run_faces.bat)
#  All arguments are passed to src/find_faces.py as-is, e.g.:
#    ./run_faces.sh --search-dir /path/pics --target-dir /path/pics/person --samples "a.jpg,b.jpg"
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$SCRIPT_DIR/face_env/bin/python" ]]; then
    echo "[ERROR] Environment not found. Run ./FaceRecognitor_portable.sh first."
    exit 1
fi

# isolation: caches inside project, GPU device 0
export UV_CACHE_DIR="$SCRIPT_DIR/.cache/uv"
export HF_HOME="$SCRIPT_DIR/.cache/huggingface"
export TMPDIR="$SCRIPT_DIR/.cache/tmp"
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
mkdir -p "$TMPDIR" 2>/dev/null

# CUDA 12 shared libs from torch / nvidia-*-cu12 pip packages
SITE="$SCRIPT_DIR/face_env/lib/python3.12/site-packages"
for d in "$SITE/torch/lib" \
         "$SITE/nvidia/cuda_runtime/lib" \
         "$SITE/nvidia/cublas/lib" \
         "$SITE/nvidia/cufft/lib" \
         "$SITE/nvidia/cudnn/lib"; do
    [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

exec "$SCRIPT_DIR/face_env/bin/python" "$SCRIPT_DIR/src/find_faces.py" "$@"
