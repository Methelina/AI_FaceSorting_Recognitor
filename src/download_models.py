#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""download_models.py — установить/обновить модели insightface.

Паттерн opera-proxy.bat:
  - локальная версия хранится в manifest.json рядом с моделями
  - удалённая версия берётся парсером GitHub API (релизы deepinsight)
  - совпало -> up-to-date, ничего не качается (идемпотентно)
  - новая версия -> качается в .part, распаковывается во временную папку,
    валидируется, и только потом старая заменяется
  - нет интернета, но модели есть -> офлайн-режим, работаем с локальными
  - нет интернета и моделей -> ошибка

pycurl обязателен: его ставит FaceRecognitor_portable(.bat/.sh/.ps1) в face_env.

Запуск:  face_env\\Scripts\\python.exe download_models.py [--model buffalo_l] [--force]
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
# запасные зеркала, если парсер (API и jina) не смог получить ссылки.
# Заполняются фактически проверенными на текущий момент ссылками
# (HEAD 200, размер совпал с GitHub API, 2026-11-11).
FALLBACK_MIRRORS = {
    "buffalo_l": [
        # свежий тег model-zoo (парсер отдаёт его первым)
        {"tag": "model-zoo", "size": 288621354,
         "url": "https://github.com/deepinsight/insightface/releases/"
                "download/model-zoo/buffalo_l.zip"},
        # релиз v0.7 (классический тег, живёт с 2023 года)
        {"tag": "v0.7", "size": 288621354,
         "url": "https://github.com/deepinsight/insightface/releases/"
                "download/v0.7/buffalo_l.zip"},
    ],
}
# ПРИМЕЧАНИЕ: подписанные release-assets.githubusercontent.com ссылки
# в фолбэк не кладутся — у них ограниченный срок жизни (X-Amz-Expires).
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
    """pycurl с ретраями, докачкой и детектором зависания; urllib fallback."""
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
        c.setopt(pycurl.LOW_SPEED_LIMIT, 1024)   # B/s — детектор зависания
        c.setopt(pycurl.LOW_SPEED_TIME, 60)      # молчит 60 c — обрыв
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


# ------------------------------------------------------------- парсер ------
def resolve_asset_api(model: str) -> dict:
    """Уровень 1: GitHub API (свежий недraft-релиз с asset <model>.zip)."""
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
                print(f"[PARSE/API] найдено в релизе '{info['tag']}': "
                      f"{info['url']}")
                return info
    raise LookupError(f"asset '{wanted}' не найден ни в одном релизе")


def resolve_asset_jina(model: str) -> dict:
    """Уровень 2: страница релизов через r.jina.ai (обход блокировок).

    Регэкспом вытаскиваем прямые ссылки download/<tag>/<model>.zip;
    берём первый match (jina отдаёт свежие релизы первыми).
    Размер через jina неизвестен — валидация только по testzip.
    """
    import re
    page = http_get(JINA_RELEASES).decode("utf-8", "replace")
    pat = re.compile(
        r"https://github\.com/deepinsight/insightface/releases/"
        r"download/([^/\s\"')]+)/" + re.escape(model) + r"\.zip")
    m = pat.search(page)
    if not m:
        raise LookupError(f"ссылка на {model}.zip не найдена на странице")
    info = {
        "tag": m.group(1),
        "url": m.group(0),
        "size": 0,
    }
    print(f"[PARSE/JINA] найдено в релизе '{info['tag']}': {info['url']}")
    return info


def resolve_asset(model: str) -> dict:
    """Парсер ссылок: GitHub API -> r.jina.ai -> хардкоженые зеркала."""
    errors = []
    for fetcher in (resolve_asset_api, resolve_asset_jina):
        try:
            return fetcher(model)
        except Exception as e:
            errors.append(f"{fetcher.__name__}: {e}")
    mirrors = FALLBACK_MIRRORS.get(model, [])
    if mirrors:
        print(f"[PARSE/FALLBACK] парсеры не сработали "
              f"({len(errors)} ошибок) — пробуем {len(mirrors)} "
              f"хардкоженых зеркал по очереди")
        return mirrors[0]
    raise LookupError("все источники не сработали: " + "; ".join(errors))


def download_with_fallback(model: str, target: str, remote: dict | None) -> dict:
    """Качает в staging-папку РЯДОМ с target (не внутри!), при неудаче
    перебирает зеркала фолбэка. Валидирует zip. Возвращает источник."""
    try:
        remote = remote or resolve_asset(model)
    except Exception as e:
        sys.exit(f"[ERROR] ссылки не получены: {e}")
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
                    raise OSError("битый файл внутри архива")
            src = dict(src, size=os.path.getsize(part))
            # распаковка в staging: rename(target->old) потом вообще не
            # трогает свежескачанные файлы, и WinError 5 не возникает
            if os.path.exists(staging):
                shutil.rmtree(staging, ignore_errors=True)
            with zipfile.ZipFile(part) as z:
                z.extractall(staging)
            if not have_models(staging):
                raise OSError("после распаковки не все onnx на месте")
            os.remove(part)
            return src
        except Exception as e:
            last_err = e
            print(f"[WARN] источник не сработал: {e}")
            # не-законченный .part оставляем для докачки, мусор удаляем
            if os.path.exists(part) and os.path.getsize(part) < 1024:
                os.remove(part)
    sys.exit(f"[ERROR] ни один источник не сработал, последний: {last_err}")


# --------------------------------------------------------------- версия ----
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


# --------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser(description="Установить/обновить модели insightface")
    ap.add_argument("--model", default="buffalo_l",
                    help="имя модели из релизов (buffalo_l / antelopev2 / ...)")
    ap.add_argument("--force", action="store_true",
                    help="перекачать даже если версия совпадает")
    args = ap.parse_args()

    # pycurl: на Windows его ставит инсталлятор; на Linux колесо может
    # не собраться без libcurl-dev — тогда graceful fallback на urllib
    global PYCURL_OK
    try:
        import pycurl  # noqa: F401
        PYCURL_OK = True
    except ImportError:
        PYCURL_OK = False
        print("[WARN] pycurl недоступен — скачивание через urllib "
              "(без ретраев/докачки)")

    target = os.path.join(MODELS_BASE, args.model)
    manifest = read_manifest(target)
    local_files_ok = have_models(target)

    # --- удалённая версия (парсер; офлайн -> работаем с локальной) -------
    remote = None
    try:
        remote = resolve_asset(args.model)
    except Exception as e:
        if local_files_ok:
            print(f"[WARN] Нет связи с GitHub ({e}). "
                  f"Офлайн-режим: работаем с локальной версией "
                  f"'{manifest.get('tag', '?')}'.")
            return
        # парсер не сработал, но фолбэк-зеркала могут качать
        if args.model not in FALLBACK_MIRRORS:
            sys.exit(f"[ERROR] Нет связи с GitHub и локальных моделей нет: {e}")

    # --- сравнение версий ------------------------------------------------
    if remote is not None:
        # size==0 (источник jina/fallback без размера) не участвует в сравнении
        same_version = (manifest.get("tag") == remote["tag"]
                        and (not remote["size"]
                             or str(manifest.get("size")) == str(remote["size"])))
        if local_files_ok and not args.force and same_version:
            print(f"[OK] {args.model} '{remote['tag']}' уже установлен "
                  f"(up-to-date): {target}")
            return
        if local_files_ok:
            if manifest.get("tag"):
                print(f"[INFO] Обновление: '{manifest.get('tag')}' -> "
                      f"'{remote['tag']}'")
            else:
                print("[INFO] Модели есть, но manifest отсутствует — перекачка")
        else:
            print(f"[INFO] Локальных моделей нет, ставим '{remote['tag']}'")
    elif local_files_ok and not args.force:
        print("[INFO] Версию проверить не удалось, модели есть — "
              "перезакачка не нужна (для принудительной: --force)")
        return
    else:
        print("[INFO] Ставим модели с фолбэк-зеркала")

    os.makedirs(target, exist_ok=True)

    # --- скачивание+распаковка в staging (рядом с target, не внутри) -----
    src = download_with_fallback(args.model, target, remote)
    staging = os.path.join(MODELS_BASE, args.model + ".new")

    # --- атомарная замена: только rename, свежие файлы не трогаем --------
    backup = target + ".old"
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)
    if local_files_ok:
        try:
            os.rename(target, backup)
        except PermissionError as e:
            # target залочен чужим процессом (антивирус/проводник/python)
            sys.exit(f"[ERROR] Не удалось переименовать '{target}': {e}. "
                     f"Новая версия готова в '{staging}' — закройте "
                     f"программы, держащие папку, и перезапустите.")
    try:
        os.rename(staging, target)
    except OSError as e:
        # откат
        if os.path.exists(backup):
            os.rename(backup, target)
        sys.exit(f"[ERROR] Не удалось подставить staging: {e}. Откат выполнен.")

    if not have_models(target):
        # откат
        if os.path.exists(backup):
            shutil.rmtree(target, ignore_errors=True)
            os.rename(backup, target)
        sys.exit("[ERROR] После распаковки не все onnx-файлы на месте. "
                 "Откат к старой версии.")
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)

    write_manifest(target, src)
    print(f"[OK] Установлено '{src['tag']}': {target}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
