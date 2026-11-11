#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""find_faces.py - find photos of a specific person by face samples
and move them to a separate folder.

Project: K:\\work\\AI\\AI_FaceRecognitor (portable env, see readme.md)

Run via launcher:
  K:\\work\\AI\\AI_FaceRecognitor\\run_faces.ps1 --search-dir ... --target-dir ... --samples ...

Run without arguments for the interactive menu.
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

# model lookup order: 1) --model-root  2) models\ next to project root
#                     3) ComfyUI install (read-only, if present)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def resolve_model_root(arg_value: str | None) -> str:
    candidates = []
    if arg_value:
        candidates.append(arg_value)
    # models\ in the project root (this script lives in src\)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(os.path.dirname(here), "models"))
    candidates.append(
        r"K:\work\AI\Comfy_UI_Trellis2\Reposit\ComfyUI\models\insightface")
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "models", "buffalo_l")):
            return c
    # nothing found - return default (insightface downloads models there)
    return candidates[1]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find photos of a person by face samples and move them "
                    "to a separate folder. Run without arguments for an "
                    "interactive menu.")
    p.add_argument("--search-dir", default=None,
                   help="folder to search for images")
    p.add_argument("--target-dir", default=None,
                   help="folder to move matched photos to (created)")
    p.add_argument("--samples", default=None,
                   help="face samples: comma-separated file list, or a "
                        "folder path containing sample images")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="cosine similarity threshold, 0..1 "
                        "(default 0.45; higher = stricter)")
    p.add_argument("--det-size", type=int, default=640,
                   help="detection size; 960 finds small faces in collages "
                        "but is slower (default 640)")
    p.add_argument("--copy", action="store_true",
                   help="copy instead of move")
    p.add_argument("--dry-run", action="store_true",
                   help="do not move/copy anything, only show counters")
    p.add_argument("--model-root", default=None,
                   help="insightface models folder (default: models\\ next "
                        "to the project root)")
    return p


def expand_samples(samples_arg: str, search_dir: str) -> list[str]:
    """--samples may be a folder (all images in it) or a comma-separated
    file list. Distinguished by whether the path exists as a folder."""
    if os.path.isdir(samples_arg):
        imgs = []
        for name in sorted(os.listdir(samples_arg)):
            full = os.path.join(samples_arg, name)
            if os.path.isfile(full) and \
                    os.path.splitext(name)[1].lower() in IMG_EXT:
                imgs.append(full)
        print(f"[SAMPLES] Folder mode: {len(imgs)} image(s) in "
              f"'{samples_arg}'")
        return imgs
    # comma-separated file list (relative names resolve against search-dir)
    out = []
    for s in samples_arg.split(","):
        s = s.strip().strip('"')
        if not s:
            continue
        out.append(s if os.path.isabs(s) else os.path.join(search_dir, s))
    print(f"[SAMPLES] File list mode: {len(out)} path(s)")
    return out


def collect_images(search_dir: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(search_dir)):
        full = os.path.join(search_dir, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in IMG_EXT:
            out.append(full)
    return out


def ask_dir(prompt: str, must_exist: bool) -> str:
    while True:
        p = input(prompt).strip().strip('"')
        if must_exist and not os.path.isdir(p):
            print(f"  [!] Folder not found: {p} - try again.")
            continue
        return p


def interactive_menu(args) -> None:
    """English interactive menu - runs when --search-dir/--target-dir/
    --samples are not given."""
    print("=" * 60)
    print("  AI FaceRecognitor - interactive setup")
    print("=" * 60)
    if args.search_dir is None:
        args.search_dir = ask_dir(
            "1/3  Folder to SEARCH for images      : ", must_exist=True)
    if args.target_dir is None:
        args.target_dir = ask_dir(
            "2/3  Folder to MOVE matched photos to : ", must_exist=False)
    if args.samples is None:
        while True:
            s = input(
                "3/3  Face samples (folder OR comma-separated files)  : "
            ).strip().strip('"')
            if not s:
                print("  [!] Empty input - try again.")
                continue
            if os.path.isdir(s) or any(os.path.exists(
                    x if os.path.isabs(x) else os.path.join(args.search_dir, x))
                    for x in s.split(",")):
                args.samples = s
                break
            print(f"  [!] Path(s) not found: {s} - try again.")
    t = input(
        f"     Similarity threshold [{args.threshold}] (Enter = keep): "
    ).strip()
    if t:
        try:
            args.threshold = float(t)
        except ValueError:
            print(f"  [!] Not a number - keeping {args.threshold}")


def main() -> None:
    args = build_argparser().parse_args()
    if (args.search_dir is None and args.target_dir is None
            and args.samples is None):
        interactive_menu(args)
    # after the menu the arguments must be in place
    missing = [a for a, v in (("--search-dir", args.search_dir),
                              ("--target-dir", args.target_dir),
                              ("--samples", args.samples)) if not v]
    if missing:
        sys.exit("Missing required arguments: " + ", ".join(missing) +
                 ". Run without arguments for the interactive menu.")
    model_root = resolve_model_root(args.model_root)

    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    # CUDA 12 DLLs: Windows - torch\lib via add_dll_directory/PATH;
    # Linux - torch and nvidia-*-cu12 packages via LD_LIBRARY_PATH.
    # site-packages taken from sys.path - correct on any OS and venv.
    site_dirs = [p for p in sys.path
                 if p and os.path.isdir(p)
                 and os.path.basename(p) in ("site-packages", "dist-packages")]
    dll_dirs = []
    for s in site_dirs:
        dll_dirs += [os.path.join(s, "torch", "lib"),
                     os.path.join(s, "nvidia", "cuda_runtime", "lib"),
                     os.path.join(s, "nvidia", "cublas", "lib"),
                     os.path.join(s, "nvidia", "cufft", "lib"),
                     os.path.join(s, "nvidia", "cudnn", "lib")]
    # Windows-only fallback: torch from the ComfyUI install (read-only)
    dll_dirs.append(
        r"K:\work\AI\Comfy_UI_Trellis2\Reposit\comfy_env\Lib\site-packages\torch\lib")
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            if os.name == "nt":
                os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (dll_dir + os.pathsep
                                             + os.environ.get("LD_LIBRARY_PATH", ""))

    # Universal trick for both OSes: importing torch loads CUDA 12
    # libraries into the process, and onnxruntime finds them by name
    # already loaded (on Linux an LD_LIBRARY_PATH set after process start
    # does not work for dlopen dependencies).
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[WARN] torch failed to import ({e}) - "
              f"if the CUDA provider does not load, CPU will be used")

    # GPU (CUDA) if available, otherwise an honest fallback to CPU
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device = "GPU (CUDA)"
    else:
        providers = ["CPUExecutionProvider"]
        device = "CPU (onnxruntime-gpu not installed or DLLs mismatch)"
    print(f"Device: {device}")
    print(f"onnxruntime providers: {available}")
    print(f"Models: {model_root}")

    app = FaceAnalysis(name="buffalo_l", root=model_root,
                       providers=providers)
    # check that the CUDA provider actually loaded instead of silently
    # falling back to CPU
    active = app.models["det_10g"].session.get_providers()[0] \
        if "det_10g" in app.models else providers[0]
    if active != "CUDAExecutionProvider" and "CUDAExecutionProvider" in available:
        print("[WARNING] CUDA provider was listed but failed to load - "
              "running on CPU. Check the onnxruntime-gpu / torch CUDA "
              "version match (see readme).")
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    # --- face samples -> mean embedding ----------------------------------
    sample_paths = expand_samples(args.samples, args.search_dir)
    if not sample_paths:
        sys.exit("No sample images found (--samples).")

    refs = []
    for path in sample_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] unreadable sample, skipped: {path}")
            continue
        faces = app.get(img)
        if not faces:
            print(f"[WARNING] no face found in sample: {path}")
            continue
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                        (f.bbox[3] - f.bbox[1]))
        refs.append(best.normed_embedding)
        print(f"sample OK: {os.path.basename(path)} "
              f"(faces: {len(faces)}, largest taken)")
    if not refs:
        sys.exit("No face found in any sample - nothing to work with.")

    ref_mean = np.mean(np.stack(refs), axis=0)
    ref_mean /= np.linalg.norm(ref_mean)

    # --- main pass ---------------------------------------------------------
    images = collect_images(args.search_dir)
    print(f"\nImages in {args.search_dir}: {len(images)}")
    print(f"Similarity threshold: {args.threshold}\n")

    os.makedirs(args.target_dir, exist_ok=True)
    # matched -> target-dir; non-matched stay in place

    matched = rejected = noface = unreadable = moved = 0
    matched_files = []

    bar = tqdm(images, desc="Scanning", unit="img", ncols=100,
               dynamic_ncols=True)
    for path in bar:
        try:
            img = cv2.imread(path)
            if img is None:
                unreadable += 1
                continue
        except Exception:
            unreadable += 1
            continue
        try:
            faces = app.get(img)
        except Exception:
            faces = []

        if not faces:
            noface += 1
            continue

        best_sim = 0.0
        for f in faces:
            sim = float(np.dot(f.normed_embedding, ref_mean))
            best_sim = max(best_sim, sim)

        if best_sim >= args.threshold:
            matched += 1
            matched_files.append((path, best_sim))
            dst = os.path.join(args.target_dir, os.path.basename(path))
            if not args.dry_run:
                if args.copy:
                    shutil.copy2(path, dst)
                else:
                    shutil.move(path, dst)
                moved += 1
        else:
            rejected += 1
        bar.set_postfix(matched=matched, missed=rejected, no_face=noface)

    bar.close()
    print("\n========== SUMMARY ==========")
    print(f"Total checked   : {len(images)}")
    print(f"Matched         : {matched}" +
          (f" (moved: {moved})" if not args.dry_run else " (dry-run)"))
    print(f"Not matched     : {rejected}")
    print(f"No faces found  : {noface}")
    print(f"Unreadable      : {unreadable}")
    if matched_files:
        print("\nMatched (name: similarity):")
        for path, sim in matched_files:
            print(f"  {os.path.basename(path)}: {sim:.3f}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
    input("\nPress Enter to close...")
