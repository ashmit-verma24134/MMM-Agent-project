from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from html import unescape
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import httpx


BASE_DIR = Path(__file__).resolve().parent
PIPELINES_DIR = BASE_DIR / "pipelines"


def _default_workdir_root() -> Path:
    env_root = os.getenv("PIPELINE_WORKDIR")
    if env_root:
        return Path(env_root)

    # Lightning Studio commonly mounts persistent project storage here.
    lightning_root = Path("/teamspace/studios/this_studio")
    if lightning_root.exists():
        return lightning_root / ".cache"

    return Path(tempfile.gettempdir())


DEFAULT_WORKDIR = _default_workdir_root() / "deployed-meet-runs"
DEFAULT_WORKDIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = DEFAULT_WORKDIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _tail(text: str, max_lines: int = 220) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _meta_path(run_id: str) -> Path:
    return _run_dir(run_id) / "run_meta.json"


def _logs_path(run_id: str) -> Path:
    return _run_dir(run_id) / "pipeline.log"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_gdrive_file_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "drive.google.com" not in host:
        return None

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path or "")
    if m:
        return m.group(1)

    qs = parse_qs(parsed.query or "")
    if "id" in qs and qs["id"]:
        return qs["id"][0]

    return None


def _download_google_drive(url: str, out_path: Path) -> None:
    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        raise ValueError("Could not parse Google Drive file id from video_url.")

    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    def _is_html_response(resp: httpx.Response) -> bool:
        ctype = (resp.headers.get("content-type") or "").lower()
        if "html" in ctype or "text/plain" in ctype:
            return True
        head = (resp.content[:256] or b"").lower()
        return b"<html" in head or b"<!doctype html" in head

    def _write_if_file(resp: httpx.Response) -> bool:
        if _is_html_response(resp):
            return False
        if not resp.content or len(resp.content) < 1024:
            return False
        out_path.write_bytes(resp.content)
        return True

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        candidates = [
            direct_url,
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        ]
        for c in candidates:
            rr = client.get(c)
            rr.raise_for_status()
            if _write_if_file(rr):
                return

        page = client.get(f"https://drive.google.com/file/d/{file_id}/view")
        page.raise_for_status()
        html = page.text or ""

        form_action_match = re.search(r'id="download-form"[^>]*action="([^"]+)"', html)
        if form_action_match:
            action = unescape(form_action_match.group(1))
            action_url = urljoin("https://drive.google.com", action)
            params = {k: v for k, v in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html)}
            form_resp = client.get(action_url, params=params)
            form_resp.raise_for_status()
            if _write_if_file(form_resp):
                return

        link_match = re.search(r'href="(/uc\?export=download[^"]+)"', html)
        if link_match:
            href = unescape(link_match.group(1)).replace("&amp;", "&")
            link_url = urljoin("https://drive.google.com", href)
            link_resp = client.get(link_url)
            link_resp.raise_for_status()
            if _write_if_file(link_resp):
                return

        cookie_confirm = None
        for k, v in page.cookies.items():
            if str(k).startswith("download_warning"):
                cookie_confirm = v
                break
        if cookie_confirm:
            confirm_url = f"https://drive.google.com/uc?export=download&confirm={cookie_confirm}&id={file_id}"
            confirm_resp = client.get(confirm_url)
            confirm_resp.raise_for_status()
            if _write_if_file(confirm_resp):
                return

        msg = "Google Drive link did not provide a downloadable file."
        low = html.lower()
        if "you need access" in low or "request access" in low:
            msg += " File is not publicly accessible."
        elif "quota exceeded" in low or "too many users have viewed or downloaded" in low:
            msg += " File appears to be quota-limited by Google Drive."
        else:
            msg += " Use a publicly accessible direct file link or local video file upload."
        raise ValueError(msg)


def _validate_video_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ValueError(f"Input video file not found: {path}")

    size = path.stat().st_size
    if size < 1024:
        raise ValueError(f"Input file is too small to be valid media: {path} ({size} bytes)")

    try:
        head = path.read_bytes()[:4096].lower()
        if b"<html" in head or b"<!doctype html" in head or b"{\"error\"" in head:
            raise ValueError(
                "Downloaded input is not a media file (looks like HTML/JSON response). "
                "Use a direct video URL or upload a file."
            )
    except ValueError:
        raise
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if (not ok) or frame_count <= 0:
            raise ValueError(
                "Input file is not a decodable video for this runtime. "
                "Provide a valid MP4 (H.264/AAC recommended)."
            )
    except ValueError:
        raise
    except Exception:
        pass


def _resolve_python_executable(python_bin: Optional[str]) -> str:
    if python_bin:
        p = Path(python_bin).expanduser()
        if not p.exists():
            raise ValueError(f"python_bin does not exist: {p}")
        return str(p.resolve())

    candidates = [
        BASE_DIR.parent / ".venv" / "Scripts" / "python.exe",
        BASE_DIR / ".venv" / "Scripts" / "python.exe",
        BASE_DIR.parent / ".venv" / "bin" / "python",
        BASE_DIR / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    return sys.executable or os.getenv("PYTHON_BIN") or "python"


def _resolve_out_dir(out_dir: Optional[str], run_id: str) -> Path:
    if out_dir:
        p = Path(out_dir)
        if not p.is_absolute():
            p = DEFAULT_WORKDIR / p
    else:
        p = DEFAULT_WORKDIR / f"run_{run_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _build_common_args(
    *,
    video_path: Path,
    out_dir: Path,
    deepgram_model: str,
    deepgram_language: Optional[str],
    deepgram_request_timeout_sec: float,
    deepgram_connect_timeout_sec: float,
    deepgram_retries: int,
    deepgram_retry_backoff_sec: float,
    force_deepgram: bool,
    force_keyframes: bool,
    pre_roll_sec: float,
    gemini_model: str,
    similarity_threshold: float,
    temperature: float,
) -> list[str]:
    args = [
        "--video",
        str(video_path),
        "--out",
        str(out_dir),
        "--deepgram-model",
        deepgram_model,
        "--deepgram-request-timeout-sec",
        str(deepgram_request_timeout_sec),
        "--deepgram-connect-timeout-sec",
        str(deepgram_connect_timeout_sec),
        "--deepgram-retries",
        str(deepgram_retries),
        "--deepgram-retry-backoff-sec",
        str(deepgram_retry_backoff_sec),
        "--pre-roll-sec",
        str(pre_roll_sec),
        "--gemini-model",
        gemini_model,
        "--similarity-threshold",
        str(similarity_threshold),
        "--temperature",
        str(temperature),
    ]
    if deepgram_language:
        args.extend(["--deepgram-language", deepgram_language])
    if force_deepgram:
        args.append("--force-deepgram")
    if force_keyframes:
        args.append("--force-keyframes")
    return args


def _build_output_files(out_dir: Path, variant: str) -> Dict[str, str]:
    return {
        "utterances": str(out_dir / "utterances.json"),
        "keyframes_parsed": str(out_dir / "keyframes_parsed.json"),
        "keyframes_with_utterances": str(out_dir / "keyframes_with_utterances.json"),
        "final_output": str(
            out_dir / ("final_output.json" if variant == "full" else "final_output_demo_code.json")
        ),
        "final_output_condensed": str(
            out_dir / ("final_output_condensed.json" if variant == "full" else "final_output_demo_code_condensed.json")
        ),
    }


def _artifact_state(output_files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    state: Dict[str, Dict[str, Any]] = {}
    for key, p in output_files.items():
        path = Path(p)
        if path.exists():
            try:
                st = path.stat()
                state[key] = {
                    "size_bytes": int(st.st_size),
                    "mtime": float(st.st_mtime),
                }
            except Exception:
                state[key] = {"size_bytes": -1, "mtime": -1.0}
    return state


def _format_artifact_compact(state: Dict[str, Dict[str, Any]]) -> str:
    if not state:
        return "none"
    parts = []
    for k in sorted(state.keys()):
        sz = float(state[k].get("size_bytes", 0))
        parts.append(f"{k}:{sz/1024.0:.1f}KB")
    return ", ".join(parts)


def _watch_run(run_id: str, proc: subprocess.Popen, started_at: float, log_fh, heartbeat_sec: float) -> None:
    heartbeat_sec = max(2.0, float(heartbeat_sec))
    last_hb = 0.0
    last_artifact_change = started_at
    last_state: Dict[str, Dict[str, Any]] = {}

    while True:
        now = time.time()
        rc = proc.poll()

        if (now - last_hb) >= heartbeat_sec:
            try:
                meta_file = _meta_path(run_id)
                meta = _read_json(meta_file) if meta_file.exists() else {"run_id": run_id}
                out_files = meta.get("output_files", {}) or {}
                cur_state = _artifact_state(out_files)
                changed = cur_state != last_state
                if changed:
                    last_artifact_change = now
                unchanged_for = now - last_artifact_change
                elapsed = now - started_at

                log_fh.write(
                    "[runner] heartbeat "
                    f"elapsed={elapsed:.1f}s pid={proc.pid} "
                    f"artifacts={len(cur_state)}/{len(out_files)} "
                    f"changed={'yes' if changed else 'no'} "
                    f"unchanged_for={unchanged_for:.1f}s "
                    f"[{_format_artifact_compact(cur_state)}]\n"
                )
                log_fh.flush()

                meta["last_heartbeat_epoch"] = now
                meta["last_heartbeat_elapsed_sec"] = round(elapsed, 3)
                meta["artifacts_ready_count"] = len(cur_state)
                meta["artifacts_total_count"] = len(out_files)
                meta["artifacts_unchanged_for_sec"] = round(unchanged_for, 3)
                _write_json(meta_file, meta)
                last_state = cur_state
            except Exception as e:
                try:
                    log_fh.write(f"[runner] heartbeat_error: {type(e).__name__}: {e}\n")
                    log_fh.flush()
                except Exception:
                    pass
            last_hb = now

        if rc is not None:
            return_code = int(rc)
            break

        time.sleep(1.0)

    finished_at = time.time()
    try:
        meta_file = _meta_path(run_id)
        meta = _read_json(meta_file) if meta_file.exists() else {"run_id": run_id}
        meta["status"] = "succeeded" if return_code == 0 else "failed"
        meta["exit_code"] = int(return_code)
        meta["finished_at_epoch"] = finished_at
        meta["duration_sec"] = round(finished_at - started_at, 3)
        _write_json(meta_file, meta)
    except Exception as e:
        try:
            log_fh.write(f"\n[runner] failed to update metadata: {type(e).__name__}: {e}\n")
            log_fh.flush()
        except Exception:
            pass

    try:
        log_fh.write(f"\n[runner] process finished with exit_code={return_code}\n")
        log_fh.flush()
    except Exception:
        pass
    finally:
        try:
            log_fh.close()
        except Exception:
            pass


def start_run(
    *,
    variant: str,
    video_file_path: Optional[str],
    video_url: Optional[str],
    out_dir: Optional[str],
    python_bin: Optional[str],
    deepgram_model: str,
    deepgram_language: Optional[str],
    deepgram_request_timeout_sec: float,
    deepgram_connect_timeout_sec: float,
    deepgram_retries: int,
    deepgram_retry_backoff_sec: float,
    force_deepgram: bool,
    force_keyframes: bool,
    pre_roll_sec: float,
    gemini_model: str,
    similarity_threshold: float,
    temperature: float,
    log_heartbeat_sec: float = 10.0,
) -> Dict[str, Any]:
    script_name = {
        "full": "run_pipeline_all.py",
        "demo-code": "run_pipeline_demo_code.py",
    }.get(variant)
    if not script_name:
        raise ValueError("variant must be one of: full, demo-code")

    pipeline_script = PIPELINES_DIR / script_name
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Missing pipeline script: {pipeline_script}")

    run_id = uuid.uuid4().hex[:12]
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if video_file_path:
        src = Path(video_file_path).expanduser().resolve()
        if not src.exists():
            raise ValueError(f"Uploaded/local video file not found: {src}")
        dst = run_dir / f"input_{run_id}{src.suffix or '.mp4'}"
        shutil.copy2(src, dst)
        video_path = dst
    elif video_url:
        suffix = Path(video_url).suffix or ".mp4"
        video_path = run_dir / f"input_{run_id}{suffix}"
        if _extract_gdrive_file_id(video_url):
            _download_google_drive(video_url, video_path)
        else:
            with httpx.stream("GET", video_url, timeout=120.0, follow_redirects=True) as r:
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
    else:
        raise ValueError("Provide one of: video_file_path or video_url")

    _validate_video_file(video_path)
    out_path = _resolve_out_dir(out_dir, run_id)
    python_exe = _resolve_python_executable(python_bin)

    cmd = [
        python_exe,
        "-u",
        str(pipeline_script),
        "--python",
        python_exe,
        *_build_common_args(
            video_path=video_path,
            out_dir=out_path,
            deepgram_model=deepgram_model,
            deepgram_language=deepgram_language,
            deepgram_request_timeout_sec=deepgram_request_timeout_sec,
            deepgram_connect_timeout_sec=deepgram_connect_timeout_sec,
            deepgram_retries=deepgram_retries,
            deepgram_retry_backoff_sec=deepgram_retry_backoff_sec,
            force_deepgram=force_deepgram,
            force_keyframes=force_keyframes,
            pre_roll_sec=pre_roll_sec,
            gemini_model=gemini_model,
            similarity_threshold=similarity_threshold,
            temperature=temperature,
        ),
    ]

    started = time.time()
    logs_path = _logs_path(run_id)
    log_fh = open(logs_path, "a", encoding="utf-8", buffering=1)
    log_fh.write(
        f"[runner] run_id={run_id} variant={variant} started_at_epoch={started}\n"
        f"[runner] command={' '.join(cmd)}\n"
        f"[runner] cwd={PIPELINES_DIR}\n\n"
        f"[runner] heartbeat_interval_sec={log_heartbeat_sec}\n"
        f"[runner] python_unbuffered=1\n\n"
    )
    log_fh.flush()

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.Popen(
        cmd,
        cwd=str(PIPELINES_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
    )

    meta = {
        "variant": variant,
        "run_id": run_id,
        "python_executable": python_exe,
        "command": cmd,
        "status": "running",
        "exit_code": None,
        "pid": proc.pid,
        "started_at_epoch": started,
        "finished_at_epoch": None,
        "duration_sec": None,
        "out_dir": str(out_path),
        "logs_path": str(logs_path),
        "heartbeat_interval_sec": float(log_heartbeat_sec),
        "output_files": _build_output_files(out_path, variant),
    }
    _write_json(_meta_path(run_id), meta)

    watcher = threading.Thread(
        target=_watch_run,
        args=(run_id, proc, started, log_fh, float(log_heartbeat_sec)),
        daemon=True,
    )
    watcher.start()

    return {
        "run_id": run_id,
        "variant": variant,
        "status": "running",
        "python_executable": python_exe,
        "status_path": f"runs/{run_id}",
        "logs_path": f"runs/{run_id}/logs",
        "final_output_path": f"runs/{run_id}/final-output",
        "final_output_condensed_path": f"runs/{run_id}/final-output/condensed",
        "out_dir": str(out_path),
    }


def get_status(run_id: str) -> Dict[str, Any]:
    p = _meta_path(run_id)
    if not p.exists():
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    return _read_json(p)


def get_logs(run_id: str, tail_lines: int = 300) -> str:
    meta = get_status(run_id)
    p = Path(meta.get("logs_path", ""))
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="replace")
    limit = max(1, min(int(tail_lines), 5000))
    return _tail(txt, max_lines=limit)


def get_final_output(run_id: str, condensed: bool = False) -> Dict[str, Any]:
    meta = get_status(run_id)
    status = meta.get("status")
    key = "final_output_condensed" if condensed else "final_output"
    out_file = Path(meta["output_files"][key])

    if status == "running":
        return {
            "run_id": run_id,
            "status": status,
            "message": "Pipeline is still running. Check logs.",
        }
    if status == "failed":
        return {
            "run_id": run_id,
            "status": status,
            "message": "Pipeline failed. Check logs.",
        }
    if not out_file.exists():
        raise FileNotFoundError(f"Output not found: {out_file}")
    return _read_json(out_file)
