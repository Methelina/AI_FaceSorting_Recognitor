#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""find_faces.py — найти фото конкретного человека по образцам лица и перенести
их в отдельную папку.

Проект: K:\\work\\AI\\AI_FaceRecognitor (портативная среда, README.md)

Запуск через лаунчер:
  K:\\work\\AI\\AI_FaceRecognitor\\run_faces.ps1 --search-dir ... --target-dir ... --samples ...
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

# модели ищутся: 1) --model-root  2) models\ рядом со скриптом
#                3) сборка ComfyUI (только чтение, если есть)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def resolve_model_root(arg_value: str | None) -> str:
    candidates = []
    if arg_value:
        candidates.append(arg_value)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models"))
    candidates.append(
        r"K:\work\AI\Comfy_UI_Trellis2\Reposit\ComfyUI\models\insightface")
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "models", "buffalo_l")):
            return c
    # ничего не нашли — вернуть первый (insightface сам скачает модели туда)
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
                        "to the script)")
    return p


def expand_samples(samples_arg: str, search_dir: str) -> list[str]:
    """--samples может быть папкой (все картинки в ней) или списком файлов
    через запятую. Различаем по факту существования пути как папки."""
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
    # список файлов через запятую (относительные имена — от search-dir)
    out = []
    for s in samples_arg.split(","):
        s = s.strip().strip('"')
        if not s:
            continue
        out.append(s if os.path.isabs(s) else os.path.join(search_dir, s))
    print(f"[SAMPLES] File list mode: {len(out)} path(s)")
    return out


def ask_dir(prompt: str, must_exist: bool) -> str:
    while True:
        p = input(prompt).strip().strip('"')
        if must_exist and not os.path.isdir(p):
            print(f"  [!] Folder not found: {p} — try again.")
            continue
        return p


def interactive_menu(args) -> None:
    """Английское интерактивное меню — запускается, когда аргументы
    --search-dir/--target-dir/--samples не заданы."""
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
            print(f"  [!] Path(s) not found: {s} — try again.")
    t = input(
        f"     Similarity threshold [{args.threshold}] (Enter = keep): "
    ).strip()
    if t:
        try:
            args.threshold = float(t)
        except ValueError:
            print(f"  [!] Not a number - keeping {args.threshold}")


def collect_images(search_dir: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(search_dir)):
        full = os.path.join(search_dir, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in IMG_EXT:
            out.append(full)
    return out


def main() -> None:
    args = build_argparser().parse_args()
    if (args.search_dir is None and args.target_dir is None
            and args.samples is None):
        interactive_menu(args)
    # после меню аргументы обязаны быть на месте
    missing = [a for a, v in (("--search-dir", args.search_dir),
                              ("--target-dir", args.target_dir),
                              ("--samples", args.samples)) if not v]
    if missing:
        sys.exit("Missing required arguments: " + ", ".join(missing) +
                 ". Run without arguments for the interactive menu.")
    model_root = resolve_model_root(args.model_root)

    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    # CUDA 12 DLL: Windows — torch\lib через add_dll_directory/PATH;
    # Linux — torch и nvidia-*-cu12 пакеты через LD_LIBRARY_PATH.
    # site-packages берём из sys.path — путь верный на любой ОС и venv.
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
    # Windows-only fallback: torch из сборки ComfyUI (только чтение)
    dll_dirs.append(
        r"K:\work\AI\Comfy_UI_Trellis2\Reposit\comfy_env\Lib\site-packages\torch\lib")
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            if os.name == "nt":
                os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (dll_dir + os.pathsep
                                             + os.environ.get("LD_LIBRARY_PATH", ""))

    # Универсальный трюк для обеих ОС: import torch загружает CUDA 12
    # библиотеки в память процесса, и onnxruntime находит их по именам
    # уже загруженными (на Linux LD_LIBRARY_PATH, выставленный после
    # старта процесса, для dlopen-зависимостей не работает).
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[WARN] torch не импортировался ({e}) — "
              f"если CUDA-провайдер не загрузится, будет CPU")

    # GPU (CUDA) если доступен, иначе честный откат на CPU
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device = "GPU (CUDA)"
    else:
        providers = ["CPUExecutionProvider"]
        device = "CPU (onnxruntime-gpu не установлен или DLL не совпали)"
    print(f"Устройство вывода: {device}")
    print(f"Провайдеры onnxruntime: {available}")
    print(f"Модели: {model_root}")

    app = FaceAnalysis(name="buffalo_l", root=model_root,
                       providers=providers)
    # проверка, что CUDA-провайдер реально загрузился, а не упал в CPU
    active = app.models["det_10g"].session.get_providers()[0] \
        if "det_10g" in app.models else providers[0]
    if active != "CUDAExecutionProvider" and "CUDAExecutionProvider" in available:
        print("[ВНИМАНИЕ] CUDA-провайдер заявлен, но не загрузился — "
              "работает CPU. Сверьте версию onnxruntime-gpu с CUDA торов "
              "(см. инструкцию).")
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    # --- образцы лица -> средний эмбеддинг -------------------------------
    sample_paths = expand_samples(args.samples, args.search_dir)
    if not sample_paths:
        sys.exit("No sample images found (--samples).")

    refs = []
    for path in sample_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ВНИМАНИЕ] образец не читается, пропущен: {path}")
            continue
        faces = app.get(img)
        if not faces:
            print(f"[ВНИМАНИЕ] лицо в образце не найдено: {path}")
            continue
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                        (f.bbox[3] - f.bbox[1]))
        refs.append(best.normed_embedding)
        print(f"образец OK: {os.path.basename(path)} "
              f"(лиц: {len(faces)}, взято крупнейшее)")
    if not refs:
        sys.exit("Ни в одном образце не найдено лицо — работать не с чем.")

    ref_mean = np.mean(np.stack(refs), axis=0)
    ref_mean /= np.linalg.norm(ref_mean)

    # --- основной проход ---------------------------------------------------
    images = collect_images(args.search_dir)
    print(f"\nИзображений в {args.search_dir}: {len(images)}")
    print(f"Порог сходства: {args.threshold}\n")

    os.makedirs(args.target_dir, exist_ok=True)
    # совпавшие -> target-dir; НЕсовпавшие остаются на месте

    matched = rejected = noface = unreadable = moved = 0
    matched_files = []

    bar = tqdm(images, desc="Проверка", unit="img", ncols=100,
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
        bar.set_postfix(совпало=matched, мимо=rejected, без_лиц=noface)

    bar.close()
    print("\n========== ИТОГ ==========")
    print(f"Всего проверено : {len(images)}")
    print(f"Совпало         : {matched}" +
          (f" (перенесено: {moved})" if not args.dry_run else " (dry-run)"))
    print(f"Не совпало      : {rejected}")
    print(f"Лиц не найдено  : {noface}")
    print(f"Не читается     : {unreadable}")
    if matched_files:
        print("\nСовпавшие (имя: сходство):")
        for path, sim in matched_files:
            print(f"  {os.path.basename(path)}: {sim:.3f}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
    input("\nНажмите Enter для закрытия...")
