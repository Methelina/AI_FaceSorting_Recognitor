#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""download_models.py - install/update insightface models.

opera-proxy.bat pattern:
  - local version stored in manifest.json next to the models
  - remote version fetched by the GitHub API parser (deepinsight releases)
  - versions match -> up-to-date, nothing downloaded (idempotent)
  - newer version -> downloaded to .part, unpacked into a staging folder,
    validated, and only then the old one is replaced
  - no internet but models exist -> offline mode, local version is used
  - no internet and no models -> error

pycurl is mandatory on Windows: the installers put it into face_env.
On Linux the wheel may fail to build without libcurl-dev - graceful
urllib fallback in that case.

Run:  face_env\\Scripts\\python.exe src\\download_models.py [--model buffalo_l] [--force]
"""
import argparse
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODELS_BASE = os.path.join(ROOT, "models", "models")
API_RELEASES = "https://api.github.com/repos/deepinsight/insightface/releases"
JINA_RELEASES = "https://r.jina.ai/https://github.com/deepinsight/insightface/releases"
# Fallback mirrors used if the parsers (API and jina) cannot get links.
# Populated with links actually verified at the moment
# (HEAD 200, size matching the GitHub API, 2026-11-11).
FALLBACK_MIRRORS = {
    "buffalo_l": [
        # fresh model-zoo tag (the parser returns it first)
        {"tag": "model-zoo", "size": 288621354,
         "url": "https://github.com/deepinsight/insightface/releases/"
                "download/model-zoo/buffalo_l.zip"},
        # v0.7 release (classic tag, alive since 2023)
        {"tag": "v0.7", "size": 288621354,
         "url": "https://github.com/deepinsight/insightface/releases/"
                "download/v0.7/buffalo_l.zip"},
    ],
}
# NOTE: signed release-assets.githubusercontent.com links are not put into
# the fallback - they expire (X-Amz-Expires).
EXPECTED = ["det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx",
            "1k3d68.onnx", "genderage.onnx"]

# ---------------------------------------------------------------- pycurl ----
PYCURL_OK = False


def http_get(url: str, timeout: int = 60) -> bytes:
    if PYCURL_OK:
        import io
        pycurl = __import__("pycurl")
        buf = io.BytesIO()
        c = pycurl.Curl()
        c.setopt(pycurl.URL, url)
        c.setopt(pycurl.WRITEDATA, buf)
        c.setopt(pycurl.FOLLOWLOCATION, 1)
        c.setopt(pycurl.CONNECTTIMEOUT, 30)
        c.setopt(pycurl.TIMEOUT, timeout)
        c.setopt(pycurl.USERAGENT, "Mozilla/5.0")
        c.perform()
        code = c.getinfo(pycurl.RESPONSE_CODE)
        c.close()
        if code != 200:
            raise OSError(f"HTTP {code}")
        return buf.getvalue()
    from urllib.request import Request, urlopen
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def download(url: str, out_path: str) -> None:
    """pycurl with retries, resume and a stall detector; urllib fallback."""
    if not PYCURL_OK:
        urllib_download(url, out_path)
        return
    import pycurl
    resume = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    with open(out_path, "ab" if resume else "wb") as f:
        c = pycurl.Curl()
        c.setopt(pycurl.URL, url)
        c.setopt(pycurl.WRITEDATA, f)
        c.setopt(pycurl.FOLLOWLOCATION, 1)
        c.setopt(pycurl.CONNECTTIMEOUT, 30)
        c.setopt(pycurl.LOW_SPEED_LIMIT, 1024)   # B/s - stall detector
        c.setopt(pycurl.LOW_SPEED_TIME, 60)      # silent 60 s - abort
        c.setopt(pycurl.USERAGENT, "Mozilla/5.0")
        if resume:
            c.setopt(pycurl.RESUME_FROM, resume)
        c.setopt(pycurl.XFERINFOFUNCTION, lambda tot, done, _, __: (
            None if tot == 0 else
            print(f"\r    {done*100//tot}% ({done//1048576}/{tot//1048576} MB)",
                  end="", flush=True)))
        c.setopt(pycurl.NOPROGRESS, 0)
        retries = 5
        for attempt in range(1, retries + 1):
            try:
                c.perform()
                break
            except pycurl.error as e:
                if attempt == retries:
                    raise
                print(f"\n    retry {attempt}/{retries} after error: {e}",
                      flush=True)
        c.close()
    print()


def urllib_download(url: str, out_path: str) -> None:
    from urllib.request import urlopen, Request
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as r, open(out_path, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            block = r.read(256 * 1024)
            if not block:
                break
            f.write(block)
            got += len(block)
            if total:
                print(f"\r    {got*100//total}% ({got//1048576}/"
                      f"{total//1048576} MB)", end="", flush=True)
    print()


# ------------------------------------------------------------- parsers -----
def resolve_asset_api(model: str) -> dict:
    """Tier 1: GitHub API (freshest non-draft release with <model>.zip)."""
    data = json.loads(http_get(API_RELEASES).decode("utf-8"))
    wanted = model + ".zip"
    for rel in data:
        if rel.get("draft"):
            continue
        for asset in rel.get("assets", []):
            if asset.get("name") == wanted:
                info = {
                    "tag": rel["tag_name"],
                    "url": asset["browser_download_url"],
                    "size": int(asset.get("size") or 0),
                }
                print(f"[PARSE/API] found in release '{info['tag']}': "
                      f"{info['url']}")
                return info
    raise LookupError(f"asset '{wanted}' not found in any release")


def resolve_asset_jina(model: str) -> dict:
    """Tier 2: releases page via r.jina.ai (bypasses blocking).

    Regex-extract direct download/<tag>/<model>.zip links;
    first match wins (jina lists fresh releases first).
    Size is unknown via jina - validated by testzip only.
    """
    import re
    page = http_get(JINA_RELEASES).decode("utf-8", "replace")
    pat = re.compile(
        r"https://github\.com/deepinsight/insightface/releases/"
        r"download/([^/\s\"')]+)/" + re.escape(model) + r"\.zip")
    m = pat.search(page)
    if not m:
        raise LookupError(f"link to {model}.zip not found on the page")
    info = {
        "tag": m.group(1),
        "url": m.group(0),
        "size": 0,
    }
    print(f"[PARSE/JINA] found in release '{info['tag']}': {info['url']}")
    return info


def resolve_asset(model: str) -> dict:
    """Link resolution: GitHub API -> r.jina.ai -> hardcoded mirrors."""
    errors = []
    for fetcher in (resolve_asset_api, resolve_asset_jina):
        try:
            return fetcher(model)
        except Exception as e:
            errors.append(f"{fetcher.__name__}: {e}")
    mirrors = FALLBACK_MIRRORS.get(model, [])
    if mirrors:
        print(f"[PARSE/FALLBACK] parsers failed "
              f"({len(errors)} errors) - trying {len(mirrors)} "
              f"hardcoded mirrors in order")
        return mirrors[0]
    raise LookupError("all sources failed: " + "; ".join(errors))


def download_with_fallback(model: str, target: str, remote: dict | None) -> dict:
    """Downloads into a staging folder NEXT to target (not inside!),
    iterating fallback mirrors on failure. Validates the zip.
    Returns the successful source."""
    try:
        remote = remote or resolve_asset(model)
    except Exception as e:
        sys.exit(f"[ERROR] could not get links: {e}")
    sources = [remote]
    seen = {remote["url"]}
    for m in FALLBACK_MIRRORS.get(model, []):
        if m["url"] not in seen:
            sources.append(m)
            seen.add(m["url"])

    staging = os.path.join(MODELS_BASE, model + ".new")
    part = staging + ".zip.part"

    last_err = None
    for src in sources:
        try:
            print(f"[DOWN] {model} '{src['tag']}': {src['url']}")
            download(src["url"], part)
            with zipfile.ZipFile(part) as z:
                if z.testzip() is not None:
                    raise OSError("corrupted file inside the archive")
            src = dict(src, size=os.path.getsize(part))
            # unpack into staging: rename(target->old) then never touches
            # freshly downloaded files, so WinError 5 cannot happen
            if os.path.exists(staging):
                shutil.rmtree(staging, ignore_errors=True)
            with zipfile.ZipFile(part) as z:
                z.extractall(staging)
            if not have_models(staging):
                raise OSError("not all onnx files present after unpack")
            os.remove(part)
            return src
        except Exception as e:
            last_err = e
            print(f"[WARN] source failed: {e}")
            # incomplete .part is kept for resume, junk is removed
            if os.path.exists(part) and os.path.getsize(part) < 1024:
                os.remove(part)
    sys.exit(f"[ERROR] no source worked, last error: {last_err}")


# --------------------------------------------------------------- version ----
def read_manifest(model_dir: str) -> dict:
    path = os.path.join(model_dir, "manifest.json")
    if os.path.isfile(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def write_manifest(model_dir: str, info: dict) -> None:
    path = os.path.join(model_dir, "manifest.json")
    json.dump(info, open(path, "w", encoding="utf-8"), indent=2)


def have_models(model_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(model_dir, f)) for f in EXPECTED)


# ------------------------------------------------------------------ main ---
def main() -> None:
    global PYCURL_OK
    ap = argparse.ArgumentParser(description="Install/update insightface models")
    ap.add_argument("--model", default="buffalo_l",
                    help="model name from releases (buffalo_l / antelopev2 / ...)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the version matches")
    args = ap.parse_args()

    # pycurl: installed by the installer on Windows; on Linux the wheel
    # may fail to build without libcurl-dev - graceful urllib fallback
    try:
        import pycurl  # noqa: F401
        PYCURL_OK = True
    except ImportError:
        PYCURL_OK = False
        print("[WARN] pycurl unavailable - downloading via urllib "
              "(no retries/resume)")

    target = os.path.join(MODELS_BASE, args.model)
    manifest = read_manifest(target)
    local_files_ok = have_models(target)

    # --- remote version (parser; offline -> use the local one) -----------
    remote = None
    try:
        remote = resolve_asset(args.model)
    except Exception as e:
        if local_files_ok:
            print(f"[WARN] Cannot reach GitHub ({e}). "
                  f"Offline mode: using the local version "
                  f"'{manifest.get('tag', '?')}'.")
            return
        # parser failed, but fallback mirrors can still download
        if args.model not in FALLBACK_MIRRORS:
            sys.exit(f"[ERROR] No GitHub access and no local models: {e}")

    # --- version comparison ----------------------------------------------
    if remote is not None:
        # size==0 (jina/fallback source) is excluded from comparison:
        # size unknown
        same_version = (manifest.get("tag") == remote["tag"]
                        and (not remote["size"]
                             or str(manifest.get("size")) == str(remote["size"])))
        if local_files_ok and not args.force and same_version:
            print(f"[OK] {args.model} '{remote['tag']}' already installed "
                  f"(up-to-date): {target}")
            return
        if local_files_ok:
            if manifest.get("tag"):
                print(f"[INFO] Update: '{manifest.get('tag')}' -> "
                      f"'{remote['tag']}'")
            else:
                print("[INFO] Models exist but manifest is missing - re-download")
        else:
            print(f"[INFO] No local models, installing '{remote['tag']}'")
    elif local_files_ok and not args.force:
        print("[INFO] Could not check the version, models exist - "
              "no re-download needed (to force: --force)")
        return
    else:
        print("[INFO] Installing models from a fallback mirror")

    os.makedirs(target, exist_ok=True)

    # --- download+unpack into staging (next to target, not inside) -------
    src = download_with_fallback(args.model, target, remote)
    staging = os.path.join(MODELS_BASE, args.model + ".new")

    # --- atomic replacement: renames only, fresh files untouched ---------
    backup = target + ".old"
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)
    if local_files_ok:
        try:
            os.rename(target, backup)
        except PermissionError as e:
            # target is locked by another process (antivirus/explorer/python)
            sys.exit(f"[ERROR] Could not rename '{target}': {e}. "
                     f"The new version is ready in '{staging}' - close "
                     f"programs holding the folder and re-run.")
    try:
        os.rename(staging, target)
    except OSError as e:
        # rollback
        if os.path.exists(backup):
            os.rename(backup, target)
        sys.exit(f"[ERROR] Could not swap in staging: {e}. Rollback done.")

    if not have_models(target):
        # rollback
        if os.path.exists(backup):
            shutil.rmtree(target, ignore_errors=True)
            os.rename(backup, target)
        sys.exit("[ERROR] Not all onnx files present after unpacking. "
                 "Rolled back to the previous version.")
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)

    write_manifest(target, src)
    print(f"[OK] Installed '{src['tag']}': {target}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
