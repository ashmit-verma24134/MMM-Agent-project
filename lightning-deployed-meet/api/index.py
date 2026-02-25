from __future__ import annotations

import json
import os
import re
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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl


BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINES_DIR = BASE_DIR / "pipelines"
DEFAULT_WORKDIR = Path(os.getenv("PIPELINE_WORKDIR", tempfile.gettempdir())) / "deployed-meet-runs"
DEFAULT_WORKDIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = DEFAULT_WORKDIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class PipelineRequest(BaseModel):
    video_path: Optional[str] = Field(default=None, description="Absolute or server-local path to input video.")
    video_url: Optional[HttpUrl] = Field(default=None, description="Optional URL to download input video from.")
    out_dir: Optional[str] = Field(default=None, description="Optional output directory. Defaults to /tmp run folder.")

    deepgram_model: str = "nova-3"
    deepgram_language: Optional[str] = None
    deepgram_request_timeout_sec: float = 1200.0
    deepgram_connect_timeout_sec: float = 30.0
    deepgram_retries: int = 3
    deepgram_retry_backoff_sec: float = 2.0
    force_deepgram: bool = False

    force_keyframes: bool = False
    pre_roll_sec: float = 3.0
    gemini_model: str = "gemini-2.5-flash"
    similarity_threshold: float = 0.82
    temperature: float = 0.2
    python_bin: Optional[str] = Field(
        default=None,
        description="Optional Python executable path for running pipeline subprocesses.",
    )
    log_heartbeat_sec: float = Field(
        default=10.0,
        description="Seconds between heartbeat progress lines written to run logs.",
    )


app = FastAPI(title="deployed-meet", version="1.0.0")


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


def _get_meta_or_404(run_id: str) -> Dict[str, Any]:
    p = _meta_path(run_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    try:
        return _read_json(p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read run metadata: {type(e).__name__}: {e}") from e


def _resolve_video_input(req: PipelineRequest, run_id: str, run_dir: Path) -> Path:
    if req.video_path:
        p = Path(req.video_path).expanduser().resolve()
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"video_path does not exist: {p}")
        return p

    if req.video_url:
        suffix = Path(str(req.video_url)).suffix or ".mp4"
        local = run_dir / f"input_{run_id}{suffix}"
        try:
            url = str(req.video_url)
            if _extract_gdrive_file_id(url):
                _download_google_drive(url, local)
            else:
                with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
                    r.raise_for_status()
                    with open(local, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video_url: {type(e).__name__}: {e}") from e
        return local

    raise HTTPException(status_code=400, detail="Provide one of: video_path or video_url.")


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
        raise HTTPException(status_code=400, detail="Could not parse Google Drive file id from video_url.")

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

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            # Try a couple of direct download endpoints first.
            candidates = [
                direct_url,
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            ]

            for c in candidates:
                rr = client.get(c)
                rr.raise_for_status()
                if _write_if_file(rr):
                    return

            # Parse Drive HTML interstitial page and submit download form if present.
            page = client.get(f"https://drive.google.com/file/d/{file_id}/view")
            page.raise_for_status()
            html = page.text or ""

            # Pattern A: explicit download form.
            form_action_match = re.search(r'id="download-form"[^>]*action="([^"]+)"', html)
            if form_action_match:
                action = unescape(form_action_match.group(1))
                action_url = urljoin("https://drive.google.com", action)
                params = {k: v for k, v in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html)}
                form_resp = client.get(action_url, params=params)
                form_resp.raise_for_status()
                if _write_if_file(form_resp):
                    return

            # Pattern B: direct download link in page HTML.
            link_match = re.search(r'href="(/uc\?export=download[^"]+)"', html)
            if link_match:
                href = unescape(link_match.group(1)).replace("&amp;", "&")
                link_url = urljoin("https://drive.google.com", href)
                link_resp = client.get(link_url)
                link_resp.raise_for_status()
                if _write_if_file(link_resp):
                    return

            # Pattern C: download_warning cookie + confirm token flow.
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
                msg += " Use a publicly accessible direct file link or local video_path."
            raise HTTPException(status_code=400, detail=msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download Google Drive file: {type(e).__name__}: {e}") from e


def _validate_video_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail=f"Input video file not found: {path}")

    size = path.stat().st_size
    if size < 1024:
        raise HTTPException(status_code=400, detail=f"Input file is too small to be valid media: {path} ({size} bytes)")

    # Common case for bad video_url: downloaded HTML/JSON page saved as .mp4.
    try:
        head = path.read_bytes()[:4096].lower()
        if b"<html" in head or b"<!doctype html" in head or b"{\"error\"" in head:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Downloaded input is not a media file (looks like HTML/JSON response). "
                    "Use a direct video file URL or provide video_path."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        pass

    # Lightweight decode check.
    try:
        import cv2  # local import to avoid import cost at startup

        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if (not ok) or frame_count <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Input file is not a decodable video for this runtime. "
                    "Provide a valid MP4 (H.264/AAC recommended) or use a direct media URL."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        # If cv2 probing fails unexpectedly, let pipeline attempt process rather than hard-fail.
        pass


def _resolve_python_executable(req: PipelineRequest) -> str:
    if req.python_bin:
        p = Path(req.python_bin).expanduser()
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"python_bin does not exist: {p}")
        return str(p.resolve())

    # Prefer project virtualenv if available.
    candidates = [
        BASE_DIR.parent / ".venv" / "Scripts" / "python.exe",  # Windows, repo root venv
        BASE_DIR / ".venv" / "Scripts" / "python.exe",         # Windows, deployed-meet local venv
        BASE_DIR.parent / ".venv" / "bin" / "python",          # Unix, repo root venv
        BASE_DIR / ".venv" / "bin" / "python",                 # Unix, deployed-meet local venv
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    # Fallback to currently running interpreter.
    return sys.executable or os.getenv("PYTHON_BIN") or "python"


def _resolve_out_dir(req: PipelineRequest, run_id: str) -> Path:
    if req.out_dir:
        p = Path(req.out_dir)
        if not p.is_absolute():
            p = DEFAULT_WORKDIR / p
    else:
        p = DEFAULT_WORKDIR / f"run_{run_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _build_common_args(req: PipelineRequest, video_path: Path, out_dir: Path) -> list[str]:
    args = [
        "--video",
        str(video_path),
        "--out",
        str(out_dir),
        "--deepgram-model",
        req.deepgram_model,
        "--deepgram-request-timeout-sec",
        str(req.deepgram_request_timeout_sec),
        "--deepgram-connect-timeout-sec",
        str(req.deepgram_connect_timeout_sec),
        "--deepgram-retries",
        str(req.deepgram_retries),
        "--deepgram-retry-backoff-sec",
        str(req.deepgram_retry_backoff_sec),
        "--pre-roll-sec",
        str(req.pre_roll_sec),
        "--gemini-model",
        req.gemini_model,
        "--similarity-threshold",
        str(req.similarity_threshold),
        "--temperature",
        str(req.temperature),
    ]
    if req.deepgram_language:
        args.extend(["--deepgram-language", req.deepgram_language])
    if req.force_deepgram:
        args.append("--force-deepgram")
    if req.force_keyframes:
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


def _watch_run(
    run_id: str,
    proc: subprocess.Popen,
    started_at: float,
    log_fh,
    heartbeat_sec: float,
) -> None:
    heartbeat_sec = max(2.0, float(heartbeat_sec))
    last_hb = 0.0
    last_artifact_change = started_at
    last_state: Dict[str, Dict[str, Any]] = {}

    # Emit periodic progress so logs are not "stuck" during long calls.
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


def _start_pipeline(pipeline_script: Path, req: PipelineRequest, variant: str) -> Dict[str, Any]:
    if not pipeline_script.exists():
        raise HTTPException(status_code=500, detail=f"Missing pipeline script: {pipeline_script}")

    run_id = uuid.uuid4().hex[:12]
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = _resolve_video_input(req, run_id, run_dir)
    _validate_video_file(video_path)
    out_dir = _resolve_out_dir(req, run_id)
    python_exe = _resolve_python_executable(req)

    cmd = [
        python_exe,
        "-u",
        str(pipeline_script),
        "--python",
        python_exe,
        *_build_common_args(req, video_path, out_dir),
    ]

    started = time.time()
    logs_path = _logs_path(run_id)
    log_fh = open(logs_path, "a", encoding="utf-8", buffering=1)
    log_fh.write(
        f"[runner] run_id={run_id} variant={variant} started_at_epoch={started}\n"
        f"[runner] command={' '.join(cmd)}\n"
        f"[runner] cwd={PIPELINES_DIR}\n\n"
        f"[runner] heartbeat_interval_sec={req.log_heartbeat_sec}\n"
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
        "out_dir": str(out_dir),
        "logs_path": str(logs_path),
        "heartbeat_interval_sec": float(req.log_heartbeat_sec),
        "output_files": _build_output_files(out_dir, variant),
    }
    _write_json(_meta_path(run_id), meta)

    watcher = threading.Thread(
        target=_watch_run,
        args=(run_id, proc, started, log_fh, float(req.log_heartbeat_sec)),
        daemon=True,
    )
    watcher.start()

    return {
        "run_id": run_id,
        "variant": variant,
        "status": "running",
        "python_executable": python_exe,
        "status_path": f"/runs/{run_id}",
        "logs_path": f"/runs/{run_id}/logs",
        "final_output_path": f"/runs/{run_id}/final-output",
        "final_output_condensed_path": f"/runs/{run_id}/final-output/condensed",
        "out_dir": str(out_dir),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "deployed-meet",
        "status": "ok",
        "docs": "/docs",
        "routes": [
            "/pipeline/full",
            "/pipeline/demo-code",
            "/runs/{run_id}",
            "/runs/{run_id}/logs",
            "/runs/{run_id}/final-output",
            "/runs/{run_id}/final-output/condensed",
        ],
    }


@app.post("/pipeline/full")
def pipeline_full(req: PipelineRequest) -> Dict[str, Any]:
    return _start_pipeline(PIPELINES_DIR / "run_pipeline_all.py", req, variant="full")


@app.post("/pipeline/demo-code")
def pipeline_demo_code(req: PipelineRequest) -> Dict[str, Any]:
    return _start_pipeline(PIPELINES_DIR / "run_pipeline_demo_code.py", req, variant="demo_code")


@app.get("/runs/{run_id}")
def run_status(run_id: str) -> Dict[str, Any]:
    return _get_meta_or_404(run_id)


@app.get("/runs/{run_id}/logs")
def run_logs(run_id: str, tail_lines: int = 300) -> PlainTextResponse:
    meta = _get_meta_or_404(run_id)
    p = Path(meta.get("logs_path", ""))
    if not p.exists():
        return PlainTextResponse("")
    txt = p.read_text(encoding="utf-8", errors="replace")
    limit = max(1, min(int(tail_lines), 5000))
    return PlainTextResponse(_tail(txt, max_lines=limit))


@app.get("/runs/{run_id}/final-output")
def run_final_output(run_id: str) -> Any:
    meta = _get_meta_or_404(run_id)
    status = meta.get("status")
    out_file = Path(meta["output_files"]["final_output"])

    if status == "running":
        return JSONResponse(
            status_code=202,
            content={
                "run_id": run_id,
                "status": status,
                "message": "Pipeline is still running. Check /runs/{run_id}/logs for live progress.",
                "logs_path": f"/runs/{run_id}/logs",
            },
        )
    if status == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "run_id": run_id,
                "status": status,
                "message": "Pipeline failed. Check logs for details.",
                "logs_path": f"/runs/{run_id}/logs",
            },
        )
    if not out_file.exists():
        raise HTTPException(status_code=404, detail=f"Final output not found: {out_file}")
    return _read_json(out_file)


@app.get("/runs/{run_id}/final-output/condensed")
def run_final_output_condensed(run_id: str) -> Any:
    meta = _get_meta_or_404(run_id)
    status = meta.get("status")
    out_file = Path(meta["output_files"]["final_output_condensed"])

    if status == "running":
        return JSONResponse(
            status_code=202,
            content={
                "run_id": run_id,
                "status": status,
                "message": "Pipeline is still running. Check /runs/{run_id}/logs for live progress.",
                "logs_path": f"/runs/{run_id}/logs",
            },
        )
    if status == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "run_id": run_id,
                "status": status,
                "message": "Pipeline failed. Check logs for details.",
                "logs_path": f"/runs/{run_id}/logs",
            },
        )
    if not out_file.exists():
        raise HTTPException(status_code=404, detail=f"Condensed final output not found: {out_file}")
    return _read_json(out_file)
