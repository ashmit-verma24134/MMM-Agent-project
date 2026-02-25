from __future__ import annotations

import inspect
import os
import socket
from pathlib import Path


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _default_workdir() -> str:
    if os.getenv("PIPELINE_WORKDIR"):
        return os.environ["PIPELINE_WORKDIR"]

    lightning_root = Path("/teamspace/studios/this_studio")
    if lightning_root.exists():
        return str(lightning_root / ".cache")
    return "/tmp"


def configure_env() -> None:
    gpu = _has_cuda()
    os.environ.setdefault("PORT", "8080")
    os.environ.setdefault("PIPELINE_WORKDIR", _default_workdir())
    os.environ.setdefault("YOLO_DEVICE", "0" if gpu else "cpu")
    os.environ.setdefault("OCR_MODE", "gpu" if gpu else "cpu")
    os.environ.setdefault("OCR_BACKEND_GPU", "easyocr")
    os.environ.setdefault("OCR_BACKEND_CPU", "rapidocr")
    os.environ.setdefault("PARSE_WORKERS", "1")
    os.environ.setdefault("GRADIO_CONCURRENCY", "1")


def _find_available_port(preferred: int, span: int = 20) -> int:
    for port in range(int(preferred), int(preferred) + int(span)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", int(port)))
            return int(port)
        except OSError:
            continue
        finally:
            sock.close()
    raise OSError(f"Cannot find empty port in range: {preferred}-{preferred + span - 1}")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    configure_env()
    preferred_port = int(os.getenv("PORT", "8080"))
    server_port = _find_available_port(preferred_port, span=50)
    os.environ["PORT"] = str(server_port)
    os.environ["GRADIO_SERVER_PORT"] = str(server_port)
    print(f"[launch_lightning] preferred_port={preferred_port} selected_port={server_port}")
    from app import demo

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": server_port,
        "show_error": True,
    }
    # Lightning proxy is more stable with non-SSR Gradio rendering.
    if "ssr_mode" in inspect.signature(demo.launch).parameters:
        launch_kwargs["ssr_mode"] = False
    if "root_path" in inspect.signature(demo.launch).parameters:
        root_path = os.getenv("GRADIO_ROOT_PATH", "").strip()
        if root_path:
            launch_kwargs["root_path"] = root_path
    if "share" in inspect.signature(demo.launch).parameters:
        launch_kwargs["share"] = _env_bool("GRADIO_SHARE", False)

    use_queue = _env_bool("GRADIO_USE_QUEUE", False)
    print(
        f"[launch_lightning] use_queue={use_queue} "
        f"share={launch_kwargs.get('share', False)} "
        f"root_path={launch_kwargs.get('root_path', '')!r}"
    )

    if use_queue:
        demo.queue(default_concurrency_limit=int(os.getenv("GRADIO_CONCURRENCY", "1"))).launch(**launch_kwargs)
    else:
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
