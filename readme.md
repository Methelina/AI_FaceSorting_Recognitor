# AI_FaceRecognitor

## What it is and who it is for

This is a program that **sorts photos by the people in them**. You show it
a few photos of a person ("samples"), point it at a folder of images — and
it scans every picture, finds faces, and moves every photo containing that
person into a separate folder.

**Who needs this:** you have 900 photos from a trip / a Pinterest board /
any folder, and you want to collect only the ones containing a specific
person — without looking through them manually.

**How to use — 4 steps** (nothing to install or configure manually, one
file does it all):

1. Run `FaceRecognitor_portable.bat` (double-click) — once; the program
   installs itself (internet needed). Wait for `INSTALLATION COMPLETE`.
2. Run `run_faces.bat` (double-click) — an English menu opens and asks 3
   questions:
   - the folder to search for photos;
   - the folder to put matches into (created for you);
   - a folder or a few sample photos of the person.
3. Press Enter — the program starts scanning, showing live progress
   (`matched / missed / no face`). About a few minutes per 1000 photos.
4. Done: the folder from step 2 now contains only photos of that person,
   and a list with similarity percentages is printed at the end.

If it moved something extra — just delete those photos from the result
folder and repeat with a stricter threshold (the program asks; higher =
stricter). If it missed some — lower it instead. Nothing is ever destroyed:
it **moves** photos, it does not delete them.

Everything else (command-line arguments, Linux, technical details) — below,
for those who need it.

---

A portable "find photos of a specific person by face samples" tool
(insightface buffalo_l / ArcFace embeddings) that moves matched files into
a separate folder.

Fully isolated and **idempotent** environment: caches, venv and models live
inside the project folder; the installer can be re-run safely — finished
stages are skipped, nothing breaks, nothing re-downloads.
**OS-agnostic, does not depend on ComfyUI**: on a clean machine you only
need cmd/shell and internet — uv downloads Python itself, everything else
is installed by the installer.

Русская версия: [readme.ru.md](readme.ru.md)

## Layout

Installers and runtimes share the same name and differ only by the format
for your OS: `FaceRecognitor_portable` — install, `run_faces` — run.

| Install | Run | OS | Notes |
|---|---|---|---|
| `FaceRecognitor_portable.bat` | `run_faces.bat` | Windows | **pure cmd**, no PowerShell required; downloads uv via curl/bitsadmin, unpacks with the built-in tar |
| `FaceRecognitor_portable.ps1` | `run_faces.ps1` | Windows (PowerShell) | full functionality |
| `FaceRecognitor_portable.sh` | `run_faces.sh` | Linux | pycurl is built from PyPI (needs libcurl-dev), otherwise falls back to urllib |

| Shared | Purpose |
|---|---|
| `download_models.py` | Model install/update, opera-proxy style: a parser chain (GitHub API → r.jina.ai → hardcoded mirrors) finds `<model>.zip` in the freshest release, versions compared via `manifest.json`, offline mode, pycurl download (retries + resume), staging extraction and atomic replacement with rollback. Standalone: `face_env\Scripts\python.exe download_models.py [--model buffalo_l] [--force]` |
| `find_faces.py` | The search/sort script itself (cross-platform; locates CUDA libraries on both Windows and Linux) |
| `wheels\pycurl-...whl` | Proven pycurl wheel for cp312 (Windows; on Linux it is built from PyPI or falls back to urllib) |
| `models\models\buffalo_l\` | insightface models (SCRFD detector + w600k_r50 recognizer) + `manifest.json` version stamp |
| `.cache\` | uv/tmp caches (safe to delete, recreated) |

Stack: uv + `face_env` venv (Python 3.12) + torch 2.8.0+cu128 (provider of
CUDA 12 DLLs for onnxruntime) + insightface 2.1 + onnxruntime-gpu **1.22.0**
(last branch built against CUDA 12) + opencv + tqdm + pycurl +
**scikit-image<0.26** (pin: insightface uses an API that breaks in
skimage 2.2).

## Install (once; re-running is safe)

```bat
FaceRecognitor_portable.bat      REM Windows (double-click)
./FaceRecognitor_portable.sh     REM Linux
```

The installer: downloads/copies uv, creates `face_env`, installs torch
cu128, insightface and the rest (each package is checked individually —
installed ones are not reinstalled), downloads the buffalo_l models and at
the end **verifies the installation itself** (imports + provider versions).

If the install is interrupted — just run it again: it resumes with whatever
is missing (including resuming an interrupted model archive).
Download models only: `face_env\Scripts\python.exe download_models.py`

## Run

**Interactive menu (English)** — just start the runtime without arguments
(double-click `run_faces.bat` / `run_faces.sh`):

```
1/3  Folder to SEARCH for images      : O:\PinterestDownloads\vip_board
2/3  Folder to MOVE matched photos to : O:\PinterestDownloads\vip_board\cailee_spaeny
3/3  Face samples (folder OR comma-separated files)  : ...
     Similarity threshold [0.45] (Enter = keep):
```

Step 3 accepts a **folder** of samples (all images in it are used) or a
**comma-separated file list** — detected automatically by whether the path
exists as a folder.

**Command-line arguments** (the same, without the menu):

```bat
run_faces.bat --search-dir O:\PinterestDownloads\vip_board --target-dir O:\PinterestDownloads\vip_board\cailee_spaeny --samples "pinterest_565905509452228620.jpg,pinterest_565905509452228613.jpg,pinterest_565905509452041855.jpg,pinterest_565905509452066315.jpg"
```

At startup it prints `Device: GPU (CUDA)` and **verifies** that the CUDA
provider actually loaded instead of silently falling back to CPU.

### Arguments

| Argument | Meaning |
|---|---|
| `--search-dir` | where to look for images (required; otherwise interactive menu) |
| `--target-dir` | where to move matches (created) (required) |
| `--samples` | face samples: a folder path OR comma-separated files (relative names resolve against search-dir) (required) |
| `--threshold 0.45` | cosine similarity threshold 0..1. Higher = stricter. Tuning: strangers in output → raise; the person is missed → lower |
| `--det-size 960` | detection size; 960 catches small faces in collages but is slower |
| `--copy` | copy instead of move |
| `--dry-run` | move nothing, only counters and the match list |
| `--model-root` | custom insightface models folder |

### tqdm counters

`matched` / `missed` / `no_face` update live; at the end — a summary
(total / matched / moved / missed / no-face / unreadable) and the list of
matched files with similarity scores. Ctrl+C interrupts: already-moved
files stay moved, the next run continues over the rest.

## Notes

- **onnxruntime-gpu strictly 1.22.x** — the last branch built against
  CUDA 12. Versions 1.23+ target CUDA 13 and are incompatible with torch
  cu128 (`cublasLt64_13.dll missing`, silent CPU fallback).
- **Model downloads** — source chain: GitHub API → r.jina.ai → hardcoded
  mirrors (model-zoo / v0.7). Every downloaded archive is validated
  (`testzip`) before the old version is replaced; replacement is atomic
  renames through a staging folder with automatic rollback on failure.
  Interrupted — re-run `download_models.py`, the download resumes.
- **GPU/CUDA on two OSes**: Windows — `torch\lib` via `add_dll_directory`;
  Linux — `nvidia-*-cu12` pip packages via `LD_LIBRARY_PATH` (set by the
  launcher before start) + early `import torch`. GPU is only needed for
  speed; CPU works too (~2 img/s).
- Samples: 3–5 clear frontal photos, varied lighting; from multi-face
  photos the largest face is taken.
- Matches are **moved** (not copied) — you can move them back manually;
  non-matches stay in place.

## Sanity check

```
run_faces.bat --help    (or run_faces.sh / run_faces.ps1)
```
First run after install — with `--dry-run` on a small folder.
