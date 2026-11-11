#!/usr/bin/env bash
# ============================================================
#  AI_FaceRecognitor - Portable Installer (Linux mirror of FaceRecognitor_portable.bat)
#  Idempotent: safe to run repeatedly. Self-contained: all caches,
#  venv and models live inside the project folder.
#  Usage: ./FaceRecognitor_portable.sh
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- isolation: no writes outside the project folder -----------------------
export UV_CACHE_DIR="$SCRIPT_DIR/.cache/uv"
export HF_HOME="$SCRIPT_DIR/.cache/huggingface"
export TMPDIR="$SCRIPT_DIR/.cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"

ENV_NAME="face_env"
PYTHON_VERSION="3.12"
UV_VERSION="0.9.14"
UV_EXE="$SCRIPT_DIR/uv"
PYTHON_EXE="$SCRIPT_DIR/$ENV_NAME/bin/python"
ARCH="$(uname -m)"

msg()  { echo " [INFO]  $1"; }
ok()   { echo " [OK]    $1"; }
warn() { echo " [WARN]  $1"; }
err()  { echo " [ERROR] $1"; }
step() { echo; echo ">>> Stage [$1/$2]: $3" >&2; }

# === 0. uv ================================================================
step 0 5 "Checking uv package manager (idempotent)..."
if [[ -x "$UV_EXE" ]]; then
    ok "uv already in project folder - skipping."
elif command -v uv >/dev/null 2>&1; then
    msg "System uv found: $(uv --version). Copying into project (portable)."
    cp "$(command -v uv)" "$UV_EXE"
else
    case "$ARCH" in
        x86_64)  UV_TARGET="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) UV_TARGET="aarch64-unknown-linux-gnu" ;;
        *) err "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    URL="https://releases.astral.sh/github/uv/releases/download/$UV_VERSION/uv-$UV_TARGET.tar.gz"
    msg "Downloading uv $UV_VERSION ($UV_TARGET)..."
    mkdir -p "$TMPDIR/uv_dl"
    if command -v curl >/dev/null 2>&1; then
        curl -L --connect-timeout 30 -o "$TMPDIR/uv.tar.gz" "$URL" || { err "download failed"; exit 1; }
    else
        wget -T 30 -O "$TMPDIR/uv.tar.gz" "$URL" || { err "download failed"; exit 1; }
    fi
    tar -xzf "$TMPDIR/uv.tar.gz" -C "$TMPDIR/uv_dl"
    find "$TMPDIR/uv_dl" -name uv -type f -exec cp {} "$UV_EXE" \; 2>/dev/null
    chmod +x "$UV_EXE"
    rm -rf "$TMPDIR/uv.tar.gz" "$TMPDIR/uv_dl"
    [[ -x "$UV_EXE" ]] || { err "uv extraction failed"; exit 1; }
fi
ok "uv ready: $("$UV_EXE" --version 2>&1 | head -1)"

# === 1. venv ==============================================================
step 1 5 "Checking Python environment ($ENV_NAME)..."
if [[ -x "$PYTHON_EXE" && -f "$SCRIPT_DIR/$ENV_NAME/pyvenv.cfg" ]]; then
    ok "$ENV_NAME already exists - skipping creation."
else
    [[ -d "$ENV_NAME" ]] && { warn "'$ENV_NAME' exists but is broken - recreating."; rm -rf "$ENV_NAME"; }
    "$UV_EXE" venv "$ENV_NAME" --python "$PYTHON_VERSION" || { err "venv creation failed"; exit 1; }
    ok "Environment created (Python $PYTHON_VERSION via uv)."
fi

uvinstall() {
    echo "   > uv pip install $1"
    "$UV_EXE" pip install --python "$PYTHON_EXE" $1
}

modcheck() {
    "$PYTHON_EXE" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('$1') else 1)" 2>/dev/null
}

# === 2. PyTorch cu128 (DLL/nvidia-lib provider for onnxruntime) ===========
step 2 5 "Checking PyTorch 2.8.0+cu128..."
if modcheck torch; then
    ok "torch already installed - skipping."
else
    # on Linux cu128 wheels pull nvidia-*-cu12 pip packages automatically -
    # they provide the shared libs onnxruntime-gpu needs (no system CUDA needed)
    uvinstall "torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128" \
        || { err "torch install failed"; exit 1; }
    ok "torch installed (nvidia cu12 libs included)."
fi

# === 3. Face recognition stack ============================================
step 3 5 "Checking insightface + onnxruntime-gpu 1.22.x + deps..."
check_install() {  # $1=pip spec  $2=import name
    if modcheck "$2"; then
        ok "$1 already installed - skipping."
    else
        uvinstall "$1" || { err "Failed: $1"; exit 1; }
    fi
}
check_install "insightface" insightface
check_install "onnxruntime-gpu==1.22.0" onnxruntime
check_install "opencv-python" cv2
check_install "tqdm" tqdm
check_install "numpy" numpy
# insightface 2.1 uses the deprecated skimage Transform.estimate API:
# FutureWarning on scikit-image >= 0.26, hard break on 2.2 - pin below
check_install '"scikit-image<0.26"' skimage
# pycurl: Linux needs libcurl-dev to build; graceful - downloader falls back to urllib
if modcheck curl; then
    ok "pycurl already installed - skipping."
else
    uvinstall "pycurl" || warn "pycurl build failed (need libcurl-dev/g++). Downloader will use urllib fallback."
fi

# === 4. InsightFace models (src/download_models.py is idempotent) =============
step 4 5 "Checking insightface models buffalo_l (download if missing)..."
"$PYTHON_EXE" "$SCRIPT_DIR/src/download_models.py" || { err "model download failed - rerun later"; exit 1; }

# === 5. Verify (cheap: imports only, no model load) =======================
step 5 5 "Verifying installation..."
"$PYTHON_EXE" - <<'EOF' || { err "Verification failed."; exit 1; }
import importlib.util as u
mods = ["torch", "insightface", "onnxruntime", "cv2", "tqdm", "numpy", "curl"]
missing = [m for m in mods if u.find_spec(m) is None]
if missing:
    print("MISSING:", ", ".join(missing)); raise SystemExit(1)
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
import onnxruntime as ort
print("onnxruntime:", ort.__version__, "| providers:", ort.get_available_providers())
print("ALL OK")
EOF

echo
echo " =========================================="
echo "   INSTALLATION COMPLETE (or already up to date)"
echo "   Run:  ./run_faces.sh --help"
echo " =========================================="
