# smart_keyframes_and_classify.py
import argparse
import json
import os
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import concurrent.futures as cf

import cv2
import numpy as np
from dotenv import load_dotenv

try:
    import clip
    import torch
    from PIL import Image
except Exception:
    clip = None
    torch = None
    Image = None

# Local models (layout + OCR)
# pip install ultralytics paddleocr paddlepaddle opencv-python numpy python-dotenv
from ultralytics import YOLO

# Avoid oneDNN fused-conv issues seen in some Paddle/PaddleOCR builds on CPU.
# Use hard overrides (not setdefault) so shell/.env values cannot re-enable it.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

# Compatibility patch for NumPy>=2 with imgaug (transitive dep of PaddleOCR).
# imgaug expects np.sctypes, removed in NumPy 2.0.
if not hasattr(np, "sctypes"):
    def _np_type(name: str, default):
        return getattr(np, name, default)

    np.sctypes = {
        "int": [_np_type("int8", int), _np_type("int16", int), _np_type("int32", int), _np_type("int64", int)],
        "uint": [_np_type("uint8", int), _np_type("uint16", int), _np_type("uint32", int), _np_type("uint64", int)],
        "float": [_np_type("float16", float), _np_type("float32", float), _np_type("float64", float)],
        "complex": [_np_type("complex64", complex), _np_type("complex128", complex)],
        "others": [_np_type("bool_", bool), _np_type("object_", object), _np_type("str_", str), _np_type("bytes_", bytes)],
    }

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

try:
    import easyocr
except Exception:
    easyocr = None


# ============================================================
# EDIT THESE IN CODE (no tuning args needed in the command)
# ============================================================

# Candidate sampling (local, no API)
SAMPLE_FPS = 1.0
RESIZE_W = 360
CANDIDATE_PERCENTILE = 70.0
MAX_CANDIDATES = 180

# Final cap
MAX_FRAMES = 150

# Fast/parse resize for local inference (CLIP)
FAST_FRAME_MAX_W = 720

# Parallelism removed (no LLM calls)
BASE_SLEEP_SEC = 0.0

# Local screen parsing (required)
ENABLE_LOCAL_SCREEN_PARSE = True

# Layout detector weights (DocLayNet-style YOLO weights recommended)
# Example: models/yolov8n-doclaynet.pt
LAYOUT_YOLO_WEIGHTS = os.getenv("LAYOUT_YOLO_WEIGHTS", "models/yolov8x-doclaynet.pt")
LAYOUT_CONF = float(os.getenv("LAYOUT_CONF", "0.25"))
LAYOUT_IOU = float(os.getenv("LAYOUT_IOU", "0.45"))

# YOLO runtime settings (GPU)
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "0")  # "0" for GPU 0, "cpu" to force CPU
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))  # try 512 for more speed if acceptable

# OCR
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.45"))

# OCR runtime settings
# Two explicit modes:
# - cpu: prioritizes speed with RapidOCR (best CPU throughput for this pipeline)
# - gpu: prioritizes accuracy on dense/code text with EasyOCR on CUDA
OCR_MODE = os.getenv("OCR_MODE", "gpu").strip().lower()
OCR_BACKEND_CPU = os.getenv("OCR_BACKEND_CPU", "rapidocr").strip().lower()
OCR_BACKEND_GPU = os.getenv("OCR_BACKEND_GPU", "easyocr").strip().lower()

# Selected at runtime in main()
OCR_BACKEND = "paddle"
USE_GPU = False
OCR_CROP_MAX_REGIONS = int(os.getenv("OCR_CROP_MAX_REGIONS", "10"))

# OCR crop scaling by frame type.
# Higher values (>=1.0) preserve/enhance tiny text, useful for code and code-like slides.
OCR_CROP_SCALE_BY_TYPE = {
    "slides": float(os.getenv("OCR_CROP_SCALE_SLIDES", "1.15")),
    "demo":   float(os.getenv("OCR_CROP_SCALE_DEMO", "1.10")),
    "code":   float(os.getenv("OCR_CROP_SCALE_CODE", "1.35")),
    "none":   float(os.getenv("OCR_CROP_SCALE_NONE", "1.10")),
}

# Global + per-type upscaling before OCR parse pass.
# Final OCR upscale = OCR_GLOBAL_UPSCALE * OCR_FRAME_UPSCALE_BY_TYPE[frame_type]
OCR_GLOBAL_UPSCALE = float(os.getenv("OCR_GLOBAL_UPSCALE", "1.25"))
OCR_FRAME_UPSCALE_BY_TYPE = {
    "slides": float(os.getenv("OCR_FRAME_UPSCALE_SLIDES", "1.10")),
    "demo":   float(os.getenv("OCR_FRAME_UPSCALE_DEMO", "1.10")),
    "code":   float(os.getenv("OCR_FRAME_UPSCALE_CODE", "1.20")),
    "none":   float(os.getenv("OCR_FRAME_UPSCALE_NONE", "1.10")),
}

# Resize input frame for YOLO/regular parse path (keep old lower-quality defaults).
PARSE_MAX_W_BY_TYPE = {
    "slides": int(os.getenv("PARSE_MAX_W_SLIDES", "1280")),
    "demo":   int(os.getenv("PARSE_MAX_W_DEMO", "1280")),
    "none":   int(os.getenv("PARSE_MAX_W_NONE", "1280")),
    "code":   int(os.getenv("PARSE_MAX_W_CODE", "99999")),  # effectively "no resize"
}

# Dedicated OCR save/parse representation (max quality for OCR only).
OCR_SAVE_MAX_W_BY_TYPE = {
    "slides": int(os.getenv("OCR_SAVE_MAX_W_SLIDES", "99999")),
    "demo":   int(os.getenv("OCR_SAVE_MAX_W_DEMO", "99999")),
    "none":   int(os.getenv("OCR_SAVE_MAX_W_NONE", "99999")),
    "code":   int(os.getenv("OCR_SAVE_MAX_W_CODE", "99999")),
}

# Selected keyframe image save quality.
KEYFRAME_JPEG_QUALITY = int(os.getenv("KEYFRAME_JPEG_QUALITY", "100"))
KEYFRAME_PNG_COMPRESSION = int(os.getenv("KEYFRAME_PNG_COMPRESSION", "0"))
KEYFRAME_IMAGE_FORMAT = os.getenv("KEYFRAME_IMAGE_FORMAT", "png").strip().lower()
if KEYFRAME_IMAGE_FORMAT not in {"png", "jpg", "jpeg"}:
    KEYFRAME_IMAGE_FORMAT = "png"

# CLIP frame type classifier
# -----------------------------
# CLIP setup (more robust, fewer “code” false-positives)
# Strategy:
# 1) Use multiple POS prompts per class (ensembling)
# 2) Add NEG prompts per class (especially for "code") and score = mean(pos) - mean(neg)
# This makes "slides with code screenshots" stay as slides, and prevents "demo with code words" -> code.
# -----------------------------

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")

# class labels (keep as-is)
CLIP_CLASS_LABELS = ["slides", "code", "demo", "none"]

# scoring mode used by your classifier code (implement if you haven't):
# score(class) = mean(sim(image, pos_prompts)) - mean(sim(image, neg_prompts))
CLIP_SCORE_MODE = os.getenv("CLIP_SCORE_MODE", "pos_minus_neg")

# If your pipeline supports a minimum margin between top-1 and top-2 to accept the prediction:
# (helps when frames are ambiguous)
CLIP_MIN_MARGIN = float(os.getenv("CLIP_MIN_MARGIN", "0.03"))

# Prompt bank: POS and NEG per class
CLIP_PROMPT_BANK = {
    "slides": {
        "pos": [
            "a screenshot of a presentation slide (PowerPoint or Google Slides)",
            "a slide with a large title at the top and bullet points below",
            "a slide canvas with wide margins and centered content",
            "a lecture slide with sections, headings, and bullet lists",
            "a slide that may include a small embedded screenshot (code or UI) but is still a slide",
            "a shared slide deck page in a video meeting (16:9 slide layout)",
        ],
        "neg": [
            "a full screen web application dashboard with navigation sidebar",
            "a desktop application interface with many clickable controls",
            "a full screen code editor filling the screen",
            "a terminal window filling the screen",
            "a webcam grid of meeting participants",
        ],
    },

    "code": {
        "pos": [
            "a full screen code editor filling most of the screen with many lines of code",
            "an IDE with syntax highlighting and line numbers, code dominates the screen",
            "a programming editor with file tree sidebar and editor pane, not inside a slide",
            "a terminal and code editor side by side with readable code dominating",
        ],
        "neg": [
            "a presentation slide that contains a screenshot of code",
            "a slide with a code snippet as part of a slide deck",
            "a slide with a code image and slide title and bullets",
            "a demo UI screen that contains a small code panel",
        ],
    },

    "demo": {
        "pos": [
            "a web application dashboard with a left navigation sidebar and multiple panels",
            "a product user interface with buttons, menus, input fields, and toolbars",
            "a browser-based app with tabs, filters, tables, charts, and navigation",
            "a desktop software UI with controls, forms, and interactive elements",
            "a product demo screen where the interface fills the screen (not a slide canvas)",
        ],
        "neg": [
            "a PowerPoint or Google Slides presentation slide",
            "a slide with title at top and bullet points",
            "a slide deck page with large margins and a single canvas",
            "a slide with an embedded screenshot of a UI",
            "a slide with a cursor hovering over a tab",
            "a slide with a code snippet or code screenshot",
        ],
    },

    "none": {
        "pos": [
            "a video call gallery view with participants and no shared screen",
            "a mostly blank screen or black screen",
            "a blurred transition frame with no readable content",
            "a loading screen with minimal content",
        ],
        "neg": [
            "a presentation slide",
            "a web application dashboard",
            "a full screen code editor",
        ],
    },
}

CLIP_CLASS_PROMPTS = [CLIP_PROMPT_BANK[c]["pos"] for c in CLIP_CLASS_LABELS]
CLIP_CLASS_NEG_PROMPTS = [CLIP_PROMPT_BANK[c]["neg"] for c in CLIP_CLASS_LABELS]

# Caps for JSON size
MAX_OCR_LINES = 300

# ---- NEW: hard global time gap between kept keyframes ----
MIN_KEYFRAME_GAP_SEC = 3.0

# Sensitivity rules (VISUAL ONLY)
SENS = {
    "slides": {"min_gap_sec": 1.2, "diff_mult": 1.60},
    "code":   {"min_gap_sec": 0.8, "diff_mult": 0.70},
    "demo":   {"min_gap_sec": 0.45, "diff_mult": 0.60},
    "none":   {"min_gap_sec": 0.55, "diff_mult": 0.95},
}

# Concurrent parsing workers (YOLO + OCR) for KEPT keyframes
PARSE_WORKERS = int(os.getenv("PARSE_WORKERS", "2"))


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CandidateFrame:
    t_sec: float
    frame_idx: int
    diff_score: float  # diff vs previous sampled frame (local)


# ----------------------------
# Utils
# ----------------------------

def fmt_hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _probe_video(video_path: Path) -> Tuple[float, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = float(frames / fps) if frames else 0.0
    cap.release()
    return float(fps), float(duration), int(frames)


def _mad_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _downscale_gray(frame_bgr: np.ndarray, resize_w: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    new_w = int(resize_w)
    new_h = int(h * (new_w / max(1, w)))
    small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _resize_frame_max_w(frame_bgr: np.ndarray, max_w: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr
    new_w = int(max_w)
    new_h = int(h * (new_w / w))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _single_line(s: str, max_len: int = 220) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[: max(0, max_len - 1)].rstrip() + "…"
    return s


# ----------------------------
# OCR runtime selection
# ----------------------------

def _has_cuda() -> bool:
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except Exception:
        return False


def _choose_ocr_mode(requested_mode: Optional[str]) -> str:
    mode = (requested_mode or OCR_MODE or "").strip().lower()
    if mode in {"cpu", "gpu"}:
        return mode
    return "gpu" if _has_cuda() else "cpu"


def _resolve_ocr_backend_for_mode(mode: str) -> Tuple[str, bool]:
    mode = _choose_ocr_mode(mode)
    gpu_available = _has_cuda()

    if mode == "gpu":
        candidates = [OCR_BACKEND_GPU, "easyocr", "paddle", "rapidocr"]
    else:
        candidates = [OCR_BACKEND_CPU, "rapidocr", "easyocr", "paddle"]

    seen = set()
    ordered = []
    for c in candidates:
        c = (c or "").strip().lower()
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        if c == "rapidocr" and RapidOCR is not None:
            return "rapidocr", False
        if c == "easyocr" and easyocr is not None:
            return "easyocr", bool(mode == "gpu" and gpu_available)
        if c == "paddle" and PaddleOCR is not None:
            return "paddle", bool(mode == "gpu" and gpu_available)

    raise RuntimeError(
        "No OCR backend is available. Install one of: rapidocr-onnxruntime, easyocr, paddleocr."
    )


def _init_ocr_model(backend: str, ocr_lang: str, use_gpu: bool):
    b = (backend or "").strip().lower()
    if b == "paddle":
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR backend requested but paddleocr is not installed.")
        return PaddleOCR(
            use_angle_cls=False,
            lang=ocr_lang,
            use_gpu=bool(use_gpu),
            show_log=False,
            enable_mkldnn=False,
            ir_optim=False,
        )
    if b == "easyocr":
        if easyocr is None:
            raise RuntimeError("EasyOCR backend requested but easyocr is not installed.")
        langs = [str(ocr_lang or "en")]
        return easyocr.Reader(langs, gpu=bool(use_gpu), verbose=False)
    if b == "rapidocr":
        if RapidOCR is None:
            raise RuntimeError("RapidOCR backend requested but rapidocr-onnxruntime is not installed.")
        return RapidOCR()
    raise RuntimeError(f"Unsupported OCR backend: {backend}")


# ----------------------------
# Video frame reader (single capture)
# ----------------------------

class VideoReader:
    def __init__(self, video_path: Path):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

    def read_at_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


# ----------------------------
# Local screen parse helpers (YOLO layout + PaddleOCR)
# ----------------------------

def _xyxy_to_int(xyxy):
    x1, y1, x2, y2 = xyxy
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _zone_for_box(box, W, H):
    cx, cy = _box_center(box)
    if cy < 0.18 * H:
        return "top"
    if cy > 0.85 * H:
        return "bottom"
    if cx < 0.33 * W:
        return "left"
    if cx > 0.67 * W:
        return "right"
    return "center"


def _sort_reading_order(items):
    return sorted(items, key=lambda it: (it["box"][1], it["box"][0]))


def run_layout_yolo(layout_model: YOLO, frame_bgr: np.ndarray) -> List[dict]:
    H, W = frame_bgr.shape[:2]
    res = layout_model.predict(
        source=frame_bgr,
        conf=LAYOUT_CONF,
        iou=LAYOUT_IOU,
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
        verbose=False
    )[0]

    regions = []
    names = res.names
    if res.boxes is None:
        return regions

    for b in res.boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        label = str(names.get(cls_id, f"class_{cls_id}"))
        box = _xyxy_to_int(b.xyxy[0].tolist())
        box = _clip_box(box, W, H)
        regions.append({"label": label, "conf": conf, "box": box})

    return _sort_reading_order(regions)


def _normalize_quad_to_box(quad: Any, W: int, H: int) -> Optional[Tuple[List[List[float]], List[int]]]:
    if isinstance(quad, np.ndarray):
        quad = quad.tolist()
    if not isinstance(quad, (list, tuple)) or len(quad) < 4:
        return None
    try:
        q = [[float(p[0]), float(p[1])] for p in quad[:4]]
        xs = [p[0] for p in q]
        ys = [p[1] for p in q]
        box = _clip_box([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))], W, H)
        return q, box
    except Exception:
        return None


def _extract_ocr_lines(ocr_model: Any, backend: str, frame_bgr: np.ndarray) -> List[dict]:
    H, W = frame_bgr.shape[:2]
    out: List[dict] = []
    backend = str(backend or "").strip().lower()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if backend == "paddle":
        result = ocr_model.ocr(rgb, cls=False)
        lines = result[0] if isinstance(result, list) and len(result) > 0 else []
        lines = lines or []
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            quad = line[0]
            text_conf = line[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue
            text, conf = text_conf
            normalized = _normalize_quad_to_box(quad, W, H)
            if normalized is None:
                continue
            q, box = normalized
            conf = float(conf)
            txt = _single_line(text, max_len=220)
            if conf < OCR_MIN_CONF or not txt:
                continue
            out.append({"text": txt, "conf": conf, "quad": q, "box": box})

    elif backend == "easyocr":
        lines = ocr_model.readtext(rgb, detail=1, paragraph=False) or []
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 3:
                continue
            quad, text, conf = line[0], line[1], line[2]
            normalized = _normalize_quad_to_box(quad, W, H)
            if normalized is None:
                continue
            q, box = normalized
            conf = float(conf)
            txt = _single_line(text, max_len=220)
            if conf < OCR_MIN_CONF or not txt:
                continue
            out.append({"text": txt, "conf": conf, "quad": q, "box": box})

    elif backend == "rapidocr":
        result = ocr_model(rgb)
        lines = []
        if isinstance(result, tuple) and len(result) >= 1:
            lines = result[0] or []
        elif isinstance(result, list):
            lines = result
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 3:
                continue
            quad, text, conf = line[0], line[1], line[2]
            normalized = _normalize_quad_to_box(quad, W, H)
            if normalized is None:
                continue
            q, box = normalized
            conf = float(conf)
            txt = _single_line(text, max_len=220)
            if conf < OCR_MIN_CONF or not txt:
                continue
            out.append({"text": txt, "conf": conf, "quad": q, "box": box})
    else:
        raise RuntimeError(f"Unsupported OCR backend: {backend}")

    return _sort_reading_order(out[: int(MAX_OCR_LINES)])


def _is_text_heavy_label(label: str) -> bool:
    lab = (label or "").lower()
    keys = ["title", "text", "list", "table", "header", "heading"]
    return any(k in lab for k in keys)


def _crop_and_scale(frame_bgr: np.ndarray, box: List[int], scale: float) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    if scale is None or float(scale) >= 0.999:
        return crop
    interp = cv2.INTER_CUBIC if float(scale) > 1.0 else cv2.INTER_AREA
    return cv2.resize(crop, (0, 0), fx=float(scale), fy=float(scale), interpolation=interp)


def _upscale_for_ocr(frame_bgr: np.ndarray, frame_type: str) -> Tuple[np.ndarray, float]:
    per_type = float(OCR_FRAME_UPSCALE_BY_TYPE.get(str(frame_type), 1.0))
    scale = float(OCR_GLOBAL_UPSCALE) * per_type
    scale = max(1.0, min(scale, 2.5))
    if scale <= 1.001:
        return frame_bgr, 1.0
    h, w = frame_bgr.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    up = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return up, scale


def run_ocr_full(
    ocr_model: Any,
    backend: str,
    frame_bgr: np.ndarray,
) -> List[dict]:
    return _extract_ocr_lines(ocr_model, backend, frame_bgr)


def run_ocr_on_text_regions(
    ocr_model: Any,
    backend: str,
    frame_bgr: np.ndarray,
    regions: List[dict],
    frame_type: str,
    max_regions: int = 10,
) -> List[dict]:
    """
    OCR on YOLO text-heavy regions (title/text/list/table/header).
    Crops are optionally downscaled by frame_type (slides/demo faster, code max).
    """
    H, W = frame_bgr.shape[:2]
    out: List[dict] = []

    scale = float(OCR_CROP_SCALE_BY_TYPE.get(str(frame_type), 1.00))
    text_regions = [r for r in regions if _is_text_heavy_label(r.get("label", ""))]
    text_regions = text_regions[: int(max_regions)]

    if not text_regions:
        return run_ocr_full(ocr_model, backend, frame_bgr)

    for r in text_regions:
        box = r["box"]
        x1, y1, _, _ = box
        crop = _crop_and_scale(frame_bgr, box, scale=scale)
        if crop is None or crop.size == 0:
            continue

        inv_scale = (1.0 / scale) if scale and scale > 0 else 1.0
        crop_lines = _extract_ocr_lines(ocr_model, backend, crop)

        for ln in crop_lines:
            quad_local = ln.get("quad") or []
            quad_global = []
            for p in quad_local:
                gx = float(p[0]) * inv_scale + float(x1)
                gy = float(p[1]) * inv_scale + float(y1)
                quad_global.append([gx, gy])
            normalized = _normalize_quad_to_box(quad_global, W, H)
            if normalized is None:
                continue
            q, gbox = normalized

            txt = _single_line(ln.get("text", ""), max_len=220)
            if not txt:
                continue

            out.append({
                "text": txt,
                "conf": float(ln.get("conf", 0.0)),
                "quad": q,
                "box": gbox,
                "from_region_label": r.get("label", ""),
                "from_region_box": box,
                "crop_scale": float(scale),
            })

            if len(out) >= int(MAX_OCR_LINES):
                break
        if len(out) >= int(MAX_OCR_LINES):
            break

    return _sort_reading_order(out[: int(MAX_OCR_LINES)])


def attach_zones(regions: List[dict], W: int, H: int) -> Dict[str, List[dict]]:
    zones = {"top": [], "left": [], "center": [], "right": [], "bottom": []}
    for r in regions:
        z = _zone_for_box(r["box"], W, H)
        zones[z].append(r)
    for z in zones:
        zones[z] = _sort_reading_order(zones[z])
    return zones


def guess_title(regions: List[dict], ocr_lines: List[dict]) -> str:
    title_boxes = []
    for r in regions:
        lab = r.get("label", "").lower()
        if ("title" in lab) or (lab == "title") or ("header" in lab and "page" not in lab):
            title_boxes.append(r["box"])

    def inside(line_box, region_box) -> bool:
        x1, y1, x2, y2 = line_box
        rx1, ry1, rx2, ry2 = region_box
        return (x1 >= rx1 - 3 and y1 >= ry1 - 3 and x2 <= rx2 + 3 and y2 <= ry2 + 3)

    if title_boxes:
        lines = []
        for ob in ocr_lines:
            for tb in title_boxes:
                if inside(ob["box"], tb):
                    lines.append(ob["text"])
                    break
        lines = [x for x in lines if x]
        if lines:
            return " ".join(lines[:3]).strip()

    if ocr_lines:
        return ocr_lines[0]["text"]
    return ""


def attach_ocr_to_regions(regions: List[dict], ocr_lines: List[dict], pad: int = 3) -> List[dict]:
    def inside(line_box, region_box) -> bool:
        x1, y1, x2, y2 = line_box
        rx1, ry1, rx2, ry2 = region_box
        return (x1 >= rx1 - pad and y1 >= ry1 - pad and x2 <= rx2 + pad and y2 <= ry2 + pad)

    out = []
    for r in regions:
        rb = r.get("box")
        if not rb:
            out.append(r)
            continue

        texts = []
        lines_in = []
        for ln in ocr_lines:
            lb = ln.get("box")
            if lb and inside(lb, rb):
                t = ln.get("text", "")
                if t:
                    texts.append(t)
                lines_in.append(ln)

        rr = dict(r)
        rr["text_lines"] = texts
        rr["text"] = " ".join(texts).strip()
        rr["ocr_line_count"] = len(lines_in)
        out.append(rr)

    return out


# ----------------------------
# CLIP frame type classifier (no LLM)
# ----------------------------

def init_clip_classifier() -> Tuple[Any, Any, Dict[str, Any], str]:
    """
    Builds a robust CLIP classifier with:
      - POS prompt ensembling per class
      - NEG prompt ensembling per class
      - score = mean(sim to POS) - mean(sim to NEG)
    Returns:
      clip_model, preprocess, pack, device
    where pack contains text features and metadata.
    """
    if clip is None or torch is None or Image is None:
        raise RuntimeError(
            "CLIP dependencies missing. Install torch and CLIP "
            "(e.g. pip install torch and pip install git+https://github.com/openai/CLIP.git)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"CLIP init failed for model '{CLIP_MODEL_NAME}': {type(e).__name__}: {e}") from e

    if len(CLIP_CLASS_PROMPTS) != len(CLIP_CLASS_LABELS):
        raise ValueError("CLIP_CLASS_PROMPTS must align with CLIP_CLASS_LABELS (same length).")

    if "CLIP_CLASS_NEG_PROMPTS" not in globals():
        raise ValueError("CLIP_CLASS_NEG_PROMPTS is missing. Define it (aligned with CLIP_CLASS_LABELS).")

    if len(CLIP_CLASS_NEG_PROMPTS) != len(CLIP_CLASS_LABELS):
        raise ValueError("CLIP_CLASS_NEG_PROMPTS must align with CLIP_CLASS_LABELS (same length).")

    flat_pos: List[str] = []
    pos_slices: List[Tuple[int, int]] = []
    idx = 0
    for prompts in CLIP_CLASS_PROMPTS:
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("Each entry in CLIP_CLASS_PROMPTS must be a non-empty list[str].")
        s = idx
        for p in prompts:
            if not isinstance(p, str):
                raise ValueError("All POS prompts must be strings.")
            flat_pos.append(p)
            idx += 1
        pos_slices.append((s, idx))

    flat_neg: List[str] = []
    neg_slices: List[Tuple[int, int]] = []
    idx = 0
    for prompts in CLIP_CLASS_NEG_PROMPTS:
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("Each entry in CLIP_CLASS_NEG_PROMPTS must be a non-empty list[str].")
        s = idx
        for p in prompts:
            if not isinstance(p, str):
                raise ValueError("All NEG prompts must be strings.")
            flat_neg.append(p)
            idx += 1
        neg_slices.append((s, idx))

    with torch.no_grad():
        pos_tokens = clip.tokenize(flat_pos).to(device)
        pos_feats_all = model.encode_text(pos_tokens)
        pos_feats_all = pos_feats_all / pos_feats_all.norm(dim=-1, keepdim=True)

        neg_tokens = clip.tokenize(flat_neg).to(device)
        neg_feats_all = model.encode_text(neg_tokens)
        neg_feats_all = neg_feats_all / neg_feats_all.norm(dim=-1, keepdim=True)

        pos_class_feats: List[torch.Tensor] = []
        neg_class_feats: List[torch.Tensor] = []

        for (s, e) in pos_slices:
            pos_class_feats.append(pos_feats_all[s:e])
        for (s, e) in neg_slices:
            neg_class_feats.append(neg_feats_all[s:e])

    pack = {
        "labels": CLIP_CLASS_LABELS,
        "pos_class_feats": pos_class_feats,
        "neg_class_feats": neg_class_feats,
        "score_mode": str(CLIP_SCORE_MODE),
        "min_margin": float(CLIP_MIN_MARGIN),
    }
    return model, preprocess, pack, device


def classify_frame_clip(
    *,
    frame_bgr: np.ndarray,
    clip_model: Any,
    clip_preprocess: Any,
    clip_text_features: Any,
    clip_device: str,
) -> Tuple[str, Dict[str, float]]:
    pack = clip_text_features
    labels: List[str] = pack["labels"]
    pos_class_feats: List[Any] = pack["pos_class_feats"]
    neg_class_feats: List[Any] = pack["neg_class_feats"]

    none_margin: float = float(pack.get("none_margin", 0.02))
    weak_thr: float = float(pack.get("weak_thr", 0.00))
    slide_close: float = float(pack.get("slide_close", 0.03))

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    image = clip_preprocess(img).unsqueeze(0).to(clip_device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        scores: List[float] = []
        for i in range(len(labels)):
            pos_feats = pos_class_feats[i].to(clip_device)
            neg_feats = neg_class_feats[i].to(clip_device)

            pos_sims = (img_feat @ pos_feats.T).squeeze(0)
            neg_sims = (img_feat @ neg_feats.T).squeeze(0)

            score = float(pos_sims.mean().item() - neg_sims.mean().item())
            scores.append(score)

    scores_np = np.array(scores, dtype=np.float32)
    score_map: Dict[str, float] = {labels[i]: float(scores_np[i]) for i in range(len(labels))}

    if "none" not in labels:
        best_idx = int(np.argmax(scores_np))
        pred = labels[best_idx]
        if ("slides" in labels) and (pred != "slides"):
            winner_score = float(score_map[pred])
            slides_score = float(score_map["slides"])
            if (winner_score - slides_score) < float(slide_close):
                pred = "slides"
        score_map["_slide_close"] = float(slide_close)
        return pred, score_map

    none_idx = int(labels.index("none"))
    none_score = float(scores_np[none_idx])

    non_none_idxs = [i for i, lab in enumerate(labels) if lab != "none"]
    best_non_none_idx = int(max(non_none_idxs, key=lambda i: float(scores_np[i])))
    best_non_none_label = labels[best_non_none_idx]
    best_non_none_score = float(scores_np[best_non_none_idx])

    if (none_score >= best_non_none_score + none_margin) or (best_non_none_score < weak_thr):
        pred = "none"
    else:
        pred = best_non_none_label

    if pred != "none" and ("slides" in labels) and (pred != "slides"):
        slides_score = float(score_map["slides"])
        winner_score = float(score_map[pred])
        if (winner_score - slides_score) < float(slide_close):
            pred = "slides"

    score_map["_best_non_none_score"] = float(best_non_none_score)
    score_map["_none_score"] = float(none_score)
    score_map["_none_margin"] = float(none_margin)
    score_map["_weak_thr"] = float(weak_thr)
    score_map["_best_non_none_idx"] = float(best_non_none_idx)
    score_map["_none_idx"] = float(none_idx)
    score_map["_slide_close"] = float(slide_close)
    if "slides" in labels:
        score_map["_slides_score"] = float(score_map["slides"])

    return pred, score_map


# ----------------------------
# Candidate detection (cheap, local)
# ----------------------------

def find_candidates_diff(
    video_path: Path,
    sample_fps: float,
    resize_w: int,
    candidate_percentile: float,
    max_candidates: int,
) -> Tuple[List[CandidateFrame], float]:
    fps, duration, total_frames = _probe_video(video_path)
    if duration <= 0 or total_frames <= 0:
        raise RuntimeError("Could not determine video duration/frames.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sample_fps = float(sample_fps)
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    step_frames = max(1, int(round(fps / sample_fps)))

    print(f"    [step1] video_fps={fps:.3f} duration_sec={duration:.2f} total_frames={total_frames}")
    print(f"    [step1] SAMPLE_FPS={sample_fps} -> step_frames={step_frames} (~{1.0/sample_fps:.2f}s per sample)")
    print(f"    [step1] RESIZE_W={resize_w} CANDIDATE_PERCENTILE={candidate_percentile} MAX_CANDIDATES={max_candidates}")

    candidates: List[CandidateFrame] = []
    diffs: List[float] = []

    prev_gray = None
    sampled = 0

    max_k = int((total_frames - 1) // step_frames) if total_frames > 0 else 0

    for k in range(max_k + 1):
        frame_idx = int(k * step_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        sampled += 1
        t_sec = frame_idx / fps

        gray = _downscale_gray(frame, resize_w=resize_w)
        d = 999.0 if prev_gray is None else _mad_diff(gray, prev_gray)

        candidates.append(CandidateFrame(t_sec=float(t_sec), frame_idx=int(frame_idx), diff_score=float(d)))
        diffs.append(float(d))
        prev_gray = gray

        if sampled % 300 == 0:
            print(f"    [step1] sampled={sampled} last_t={fmt_hhmmss(t_sec)} last_diff={d:.2f}")

    cap.release()

    if not candidates:
        print("    [step1] no candidates produced (empty video?)")
        return [], 0.0

    diffs_np = np.array(diffs, dtype=np.float32)
    diffs_for_thr = diffs_np[1:] if len(diffs_np) > 1 else diffs_np
    base_thr = float(np.percentile(diffs_for_thr, float(candidate_percentile)))
    base_thr = max(4.0, base_thr)

    order = np.argsort(diffs_np)[::-1]
    picked = set()
    out: List[CandidateFrame] = []

    out.append(candidates[0])
    picked.add(0)

    for idx in order:
        if len(out) >= int(max_candidates):
            break
        ii = int(idx)
        if ii in picked:
            continue
        out.append(candidates[ii])
        picked.add(ii)

    out.sort(key=lambda x: x.t_sec)

    print(f"    [step1] sampled_frames={sampled} raw_candidates={len(candidates)} selected_candidates={len(out)} base_thr={base_thr:.2f}")
    return out, base_thr


# ----------------------------
# Keyframe keep rule (visual only)
# ----------------------------

def should_keep_visual_only(
    *,
    frame_type: str,
    t_sec: float,
    diff_to_last_keep: float,
    base_thr: float,
    last_kept_t: float,
) -> Tuple[bool, Dict[str, float]]:
    cfg = SENS.get(frame_type, {"min_gap_sec": 1.0, "diff_mult": 1.0})
    diff_mult = float(cfg.get("diff_mult", 1.0))

    min_gap = float(MIN_KEYFRAME_GAP_SEC)

    ok_gap = True if last_kept_t <= -1e8 else ((t_sec - last_kept_t) >= min_gap)
    thr_eff = float(base_thr * diff_mult)
    ok_visual = diff_to_last_keep >= thr_eff

    debug = {
        "diff_to_last_keep": float(diff_to_last_keep),
        "thr_effective": float(thr_eff),
        "ok_gap": 1.0 if ok_gap else 0.0,
        "ok_visual": 1.0 if ok_visual else 0.0,
        "min_gap_sec_used": float(min_gap),
    }
    return (ok_gap and ok_visual), debug


# ----------------------------
# Concurrent parsing worker (YOLO + OCR) for kept keyframes
# ----------------------------

_WORKER_LAYOUT_MODEL = None
_WORKER_OCR_MODEL = None
_WORKER_OCR_BACKEND = "paddle"
_WORKER_OCR_USE_GPU = False

def _worker_init(
    layout_weights: str,
    ocr_lang: str,
    enable_yolo: bool = True,
    ocr_backend: str = "paddle",
    ocr_use_gpu: bool = False,
):
    global _WORKER_LAYOUT_MODEL, _WORKER_OCR_MODEL, _WORKER_OCR_BACKEND, _WORKER_OCR_USE_GPU
    _WORKER_LAYOUT_MODEL = YOLO(layout_weights) if enable_yolo else None
    _WORKER_OCR_BACKEND = str(ocr_backend or "paddle").strip().lower()
    _WORKER_OCR_USE_GPU = bool(ocr_use_gpu)
    _WORKER_OCR_MODEL = _init_ocr_model(
        backend=_WORKER_OCR_BACKEND,
        ocr_lang=ocr_lang,
        use_gpu=_WORKER_OCR_USE_GPU,
    )

def _parse_one_keyframe(job: dict) -> dict:
    global _WORKER_LAYOUT_MODEL, _WORKER_OCR_MODEL, _WORKER_OCR_BACKEND, _WORKER_OCR_USE_GPU
    kidx = int(job["keyframe_idx"])
    img_path = job["image_path"]
    frame_type = str(job.get("frame_type", "none"))
    parse_mode = str(job.get("parse_mode", "yolo_ocr"))

    frame = cv2.imread(str(img_path))
    if frame is None:
        return {"keyframe_idx": kidx, "error": f"Could not read image: {img_path}"}

    preprocessed_for_ocr = bool(job.get("ocr_preprocessed", False))
    if preprocessed_for_ocr:
        # Frame in frames_selected is already the OCR-max-quality representation.
        frame_for_ocr = frame
        ocr_parse_max_w = int(job.get("ocr_preprocess_parse_max_w", frame_for_ocr.shape[1]))
        ocr_frame_upscale_used = float(job.get("ocr_preprocess_upscale", 1.0))
    else:
        # Backward-compatible path for old runs where frames were saved unprocessed.
        ocr_parse_max_w = int(OCR_SAVE_MAX_W_BY_TYPE.get(frame_type, 99999))
        frame_for_ocr = _resize_frame_max_w(frame, max_w=ocr_parse_max_w) if ocr_parse_max_w < 99999 else frame
        frame_for_ocr, ocr_frame_upscale_used = _upscale_for_ocr(frame_for_ocr, frame_type)

    # Keep YOLO/regular parse at lower-quality defaults for speed.
    yolo_parse_max_w = int(PARSE_MAX_W_BY_TYPE.get(frame_type, 1280))
    frame_for_yolo = _resize_frame_max_w(frame, max_w=yolo_parse_max_w) if yolo_parse_max_w < 99999 else frame

    H, W = frame_for_ocr.shape[:2]

    regions: List[dict] = []
    t_yolo_ms = 0.0
    if parse_mode == "yolo_ocr":
        if _WORKER_LAYOUT_MODEL is None:
            return {"keyframe_idx": kidx, "error": "YOLO model is not initialized for yolo_ocr parse mode."}
        t0 = time.perf_counter()
        regions = run_layout_yolo(_WORKER_LAYOUT_MODEL, frame_for_yolo)
        # If OCR frame has different size, remap YOLO boxes to OCR frame coordinates.
        y_h, y_w = frame_for_yolo.shape[:2]
        o_h, o_w = frame_for_ocr.shape[:2]
        if y_h > 0 and y_w > 0 and (y_h != o_h or y_w != o_w):
            sx = float(o_w) / float(y_w)
            sy = float(o_h) / float(y_h)
            remapped = []
            for r in regions:
                bx = r.get("box")
                if not bx or len(bx) != 4:
                    continue
                x1, y1, x2, y2 = bx
                new_box = _clip_box(
                    [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))],
                    o_w,
                    o_h,
                )
                rr = dict(r)
                rr["box"] = new_box
                remapped.append(rr)
            regions = remapped
        t_yolo_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if parse_mode == "yolo_ocr":
        ocr_lines = run_ocr_on_text_regions(
            _WORKER_OCR_MODEL,
            _WORKER_OCR_BACKEND,
            frame_for_ocr,
            regions,
            frame_type=frame_type,
            max_regions=OCR_CROP_MAX_REGIONS,
        )
    else:
        # OCR-only mode: no layout detection, run full-frame OCR.
        ocr_lines = run_ocr_full(_WORKER_OCR_MODEL, _WORKER_OCR_BACKEND, frame_for_ocr)
    t_ocr_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    regions_with_text = attach_ocr_to_regions(regions, ocr_lines) if regions else []
    zones = attach_zones(regions_with_text, W=W, H=H) if regions_with_text else {"top": [], "left": [], "center": [], "right": [], "bottom": []}
    title_guess_val = guess_title(regions_with_text, ocr_lines)
    t_attach_ms = (time.perf_counter() - t0) * 1000.0

    text_lines = [x["text"] for x in ocr_lines if x.get("text")][:MAX_OCR_LINES]

    screen_parse = {
        "frame_w": int(W),
        "frame_h": int(H),
        "layout_regions": regions_with_text,
        "ocr_lines": ocr_lines,
        "zones": zones,
        "title_guess": title_guess_val,
        "layout_model": str(LAYOUT_YOLO_WEIGHTS),
        "ocr_lang": str(OCR_LANG),
        "layout_conf": float(LAYOUT_CONF),
        "layout_iou": float(LAYOUT_IOU),
        "ocr_min_conf": float(OCR_MIN_CONF),
        "parse_input_frame_type": str(frame_type),
        "yolo_device": str(YOLO_DEVICE),
        "yolo_imgsz": int(YOLO_IMGSZ),
        "ocr_backend": str(_WORKER_OCR_BACKEND),
        "ocr_use_gpu": bool(_WORKER_OCR_USE_GPU),
        "ocr_angle_cls": False,
        "ocr_crop_max_regions": int(OCR_CROP_MAX_REGIONS),
        "ocr_crop_scale_used": float(OCR_CROP_SCALE_BY_TYPE.get(frame_type, 1.00)),
        "ocr_frame_upscale_used": float(ocr_frame_upscale_used),
        "ocr_parse_max_w_used": int(ocr_parse_max_w),
        "yolo_parse_max_w_used": int(yolo_parse_max_w),
        "ocr_preprocessed_input": bool(preprocessed_for_ocr),
        "parse_mode": str(parse_mode),
    }

    return {
        "keyframe_idx": kidx,
        "on_screen_text": text_lines,
        "screen_parse": screen_parse,
        "parse_timings_ms": {
            "full_yolo_ms": float(t_yolo_ms),
            "full_ocr_ms": float(t_ocr_ms),
            "attach_text_ms": float(t_attach_ms),
        }
    }


# ----------------------------
# Main
# ----------------------------

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to meeting.mp4")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--no-yolo-for-non-demo",
        action="store_true",
        help="Use OCR-only parsing for non-demo frames (slides/code/none).",
    )
    ap.add_argument(
        "--ocr-mode",
        choices=["cpu", "gpu"],
        default=None,
        help="OCR mode. cpu = RapidOCR-first, gpu = EasyOCR-first.",
    )
    args = ap.parse_args()

    if not ENABLE_LOCAL_SCREEN_PARSE:
        raise RuntimeError("ENABLE_LOCAL_SCREEN_PARSE must be True. YOLO + OCR are required.")

    if not Path(LAYOUT_YOLO_WEIGHTS).exists():
        raise FileNotFoundError(f"Layout YOLO weights not found at: {LAYOUT_YOLO_WEIGHTS}")

    try:
        _ = YOLO(LAYOUT_YOLO_WEIGHTS)
    except Exception as e:
        raise RuntimeError(f"YOLO init failed: {type(e).__name__}: {e}") from e

    global OCR_MODE, OCR_BACKEND, USE_GPU
    OCR_MODE = _choose_ocr_mode(args.ocr_mode)
    OCR_BACKEND, USE_GPU = _resolve_ocr_backend_for_mode(OCR_MODE)

    try:
        _ = _init_ocr_model(backend=OCR_BACKEND, ocr_lang=OCR_LANG, use_gpu=USE_GPU)
    except Exception as e:
        raise RuntimeError(
            f"OCR init failed for backend={OCR_BACKEND} mode={OCR_MODE}: {type(e).__name__}: {e}"
        ) from e

    try:
        clip_model, clip_preprocess, clip_text_features, clip_device = init_clip_classifier()
    except Exception as e:
        raise RuntimeError(f"CLIP classifier init failed: {type(e).__name__}: {e}") from e

    video_path = Path(args.video).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = out_dir / "frames_selected"
    frames_dir.mkdir(parents=True, exist_ok=True)

    enriched_json = out_dir / "keyframes_parsed.json"
    timing_json = out_dir / "timing_summary.json"
    classified_dir = out_dir / "classified"
    classified_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "slides": classified_dir / "slides_keyframes.json",
        "code": classified_dir / "code_keyframes.json",
        "demo": classified_dir / "demo_keyframes.json",
        "none": classified_dir / "none_keyframes.json",
    }

    t_total0 = time.perf_counter()
    timing_totals = {
        "candidate_detection_ms": 0.0,
        "candidate_loop_ms": 0.0,
        "read_frame_ms": 0.0,
        "gray_diff_ms": 0.0,
        "clip_ms": 0.0,
        "keep_logic_ms": 0.0,
        "save_frame_ms": 0.0,
        "parse_concurrent_ms": 0.0,
        "json_write_ms": 0.0,
    }

    all_selected: List[dict] = []
    processed_times: set = set()

    last_kept_t = -1e9
    last_kept_gray: Optional[np.ndarray] = None

    if (not args.force) and enriched_json.exists():
        try:
            all_selected = safe_read_json(enriched_json)
            if isinstance(all_selected, list) and all_selected:
                processed_times = {round(float(x.get("t_sec", -1.0)), 2) for x in all_selected if "t_sec" in x}
                last = all_selected[-1]
                last_kept_t = float(last.get("t_sec", last_kept_t))

                last_img = Path(last.get("image_path", ""))
                if last_img.exists():
                    img = cv2.imread(str(last_img))
                    if img is not None:
                        last_kept_gray = _downscale_gray(img, RESIZE_W)

                print(f"Resuming: already selected {len(all_selected)} keyframes (last at {fmt_hhmmss(last_kept_t)}).")
        except Exception:
            all_selected = []
            processed_times = set()
            last_kept_t = -1e9
            last_kept_gray = None

    if args.force:
        all_selected = []
        processed_times = set()
        last_kept_t = -1e9
        last_kept_gray = None

    print("1) Finding candidate change points locally (no API)...")
    print(f"    [step1] starting... (this can take time on long videos)")
    t0 = time.perf_counter()
    candidates, base_thr = find_candidates_diff(
        video_path=video_path,
        sample_fps=SAMPLE_FPS,
        resize_w=RESIZE_W,
        candidate_percentile=CANDIDATE_PERCENTILE,
        max_candidates=MAX_CANDIDATES,
    )
    t1_ms = (time.perf_counter() - t0) * 1000.0
    timing_totals["candidate_detection_ms"] += t1_ms
    print(f"    [step1] done in {t1_ms/1000.0:.2f}s")

    print(f"Candidates: {len(candidates)}, base diff threshold ~ {base_thr:.2f}")
    print("Sensitivity config (edit in code):", SENS)
    print("Layout model:", LAYOUT_YOLO_WEIGHTS)
    print("YOLO device:", YOLO_DEVICE, "| imgsz:", YOLO_IMGSZ)
    print(
        "OCR mode:",
        OCR_MODE,
        "| backend:",
        OCR_BACKEND,
        "| lang:",
        OCR_LANG,
        "| OCR_MIN_CONF:",
        OCR_MIN_CONF,
        "| OCR GPU:",
        USE_GPU,
        "| angle_cls:",
        False,
    )
    print("OCR global upscale:", OCR_GLOBAL_UPSCALE)
    print("OCR frame upscale by type:", OCR_FRAME_UPSCALE_BY_TYPE)
    print("OCR save max_w by type:", OCR_SAVE_MAX_W_BY_TYPE)
    print("Keyframe JPEG quality:", KEYFRAME_JPEG_QUALITY)
    print("Keyframe image format:", KEYFRAME_IMAGE_FORMAT, "| PNG compression:", KEYFRAME_PNG_COMPRESSION)
    print("CLIP model:", CLIP_MODEL_NAME, "| device:", clip_device)
    print("Parse workers:", PARSE_WORKERS)
    print(f"Global min gap override (seconds since last keyframe): {MIN_KEYFRAME_GAP_SEC:.2f}s")

    kept_count = len(all_selected)
    reader = VideoReader(video_path)

    try:
        print("2) Selecting keyframes (VISUAL ONLY: time gap + diff; no OCR in loop)...")
        t_loop0 = time.perf_counter()

        for i, cand in enumerate(candidates, start=1):
            if kept_count >= int(MAX_FRAMES):
                break

            t_key = round(float(cand.t_sec), 2)
            if t_key in processed_times:
                continue
            if cand.t_sec <= (last_kept_t + 1e-6) and last_kept_t > -1e8:
                continue

            gap = float(cand.t_sec - last_kept_t) if last_kept_t > -1e8 else 9999.0

            if last_kept_t > -1e8 and gap < float(MIN_KEYFRAME_GAP_SEC):
                continue

            t0 = time.perf_counter()
            frame = reader.read_at_frame(cand.frame_idx)
            timing_totals["read_frame_ms"] += (time.perf_counter() - t0) * 1000.0
            if frame is None:
                continue

            t0 = time.perf_counter()
            gray_now = _downscale_gray(frame, RESIZE_W)
            diff_to_last_keep = 999.0 if last_kept_gray is None else _mad_diff(gray_now, last_kept_gray)
            timing_totals["gray_diff_ms"] += (time.perf_counter() - t0) * 1000.0

            print(
                f"[{i}/{len(candidates)}] t={fmt_hhmmss(cand.t_sec)} "
                f"gap_since_last_keep={gap:.2f}s cand_diff={cand.diff_score:.2f} keep_diff={diff_to_last_keep:.2f} ..."
            )

            frame_fast = _resize_frame_max_w(frame, FAST_FRAME_MAX_W)

            t0 = time.perf_counter()
            frame_type, clip_probs = classify_frame_clip(
                frame_bgr=frame_fast,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_text_features=clip_text_features,
                clip_device=clip_device,
            )
            t_clip_ms = (time.perf_counter() - t0) * 1000.0
            timing_totals["clip_ms"] += t_clip_ms

            t0 = time.perf_counter()
            keep, dbg = should_keep_visual_only(
                frame_type=frame_type,
                t_sec=float(cand.t_sec),
                diff_to_last_keep=float(diff_to_last_keep),
                base_thr=float(base_thr),
                last_kept_t=float(last_kept_t),
            )
            t_keep_ms = (time.perf_counter() - t0) * 1000.0
            timing_totals["keep_logic_ms"] += t_keep_ms

            print(
                f"    timings: clip={t_clip_ms:.0f}ms keep_logic={t_keep_ms:.0f}ms "
                f"| type={frame_type} keep={keep} | diff={diff_to_last_keep:.2f} thr_eff={dbg['thr_effective']:.2f} "
                f"| min_gap_used={dbg.get('min_gap_sec_used', MIN_KEYFRAME_GAP_SEC):.2f}s"
            )

            if not keep:
                if BASE_SLEEP_SEC > 0:
                    time.sleep(BASE_SLEEP_SEC)
                continue

            t0 = time.perf_counter()
            img_ext = ".png" if KEYFRAME_IMAGE_FORMAT == "png" else ".jpg"
            out_img = frames_dir / f"frame_{kept_count:04d}_{cand.t_sec:.2f}s_{frame_type}{img_ext}"

            # Save the exact frame representation that OCR will parse later.
            parse_max_w_for_save = int(OCR_SAVE_MAX_W_BY_TYPE.get(frame_type, 99999))
            frame_for_ocr_save = (
                _resize_frame_max_w(frame, max_w=parse_max_w_for_save)
                if parse_max_w_for_save < 99999
                else frame
            )
            frame_for_ocr_save, ocr_pre_upscale = _upscale_for_ocr(frame_for_ocr_save, frame_type)
            if KEYFRAME_IMAGE_FORMAT == "png":
                cv2.imwrite(
                    str(out_img),
                    frame_for_ocr_save,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), int(KEYFRAME_PNG_COMPRESSION)],
                )
            else:
                cv2.imwrite(
                    str(out_img),
                    frame_for_ocr_save,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(KEYFRAME_JPEG_QUALITY)],
                )
            t_save_ms = (time.perf_counter() - t0) * 1000.0
            timing_totals["save_frame_ms"] += t_save_ms

            item = {
                "keyframe_idx": int(kept_count),
                "t_sec": float(cand.t_sec),
                "timestamp": fmt_hhmmss(cand.t_sec),
                "image_path": str(out_img),

                "frame_type": frame_type,
                "on_screen_text": [],
                "screen_parse": None,
                "ocr_preprocessed": True,
                "ocr_preprocess_parse_max_w": int(parse_max_w_for_save),
                "ocr_preprocess_upscale": float(ocr_pre_upscale),

                "candidate_diff_score": float(cand.diff_score),
                "diff_to_last_keep": float(diff_to_last_keep),
                "base_diff_threshold": float(base_thr),
                "thr_effective": float(dbg.get("thr_effective", 0.0)),
                "gap_since_last_keep_sec": float(gap),

                "clip_probs": {k: float(v) for k, v in clip_probs.items()},
                "clip_prompt_map": dict(zip(CLIP_CLASS_LABELS, CLIP_CLASS_PROMPTS)),
                "clip_model_name": str(CLIP_MODEL_NAME),

                "timings_ms": {
                    "clip_ms": float(t_clip_ms),
                    "keep_logic_ms": float(t_keep_ms),
                    "save_frame_ms": float(t_save_ms),
                },
            }

            all_selected.append(item)
            processed_times.add(t_key)
            kept_count += 1

            last_kept_t = float(cand.t_sec)
            last_kept_gray = gray_now

            t0 = time.perf_counter()
            safe_write_json(enriched_json, all_selected)
            timing_totals["json_write_ms"] += (time.perf_counter() - t0) * 1000.0

            if BASE_SLEEP_SEC > 0:
                time.sleep(BASE_SLEEP_SEC)

        timing_totals["candidate_loop_ms"] += (time.perf_counter() - t_loop0) * 1000.0

    finally:
        reader.close()

    # Phase 3: YOLO + OCR concurrently on keyframes that need parsing
    to_parse = []
    for it in all_selected:
        if (not args.force) and isinstance(it.get("screen_parse"), dict) and it.get("on_screen_text"):
            continue
        if it.get("image_path"):
            frame_type = str(it.get("frame_type", "none"))
            parse_mode = "yolo_ocr"
            if args.no_yolo_for_non_demo and frame_type != "demo":
                parse_mode = "ocr_only"
            to_parse.append({
                "keyframe_idx": int(it["keyframe_idx"]),
                "t_sec": float(it["t_sec"]),
                "frame_type": frame_type,
                "image_path": str(it["image_path"]),
                "parse_mode": parse_mode,
                "ocr_preprocessed": bool(it.get("ocr_preprocessed", False)),
                "ocr_preprocess_parse_max_w": int(it.get("ocr_preprocess_parse_max_w", OCR_SAVE_MAX_W_BY_TYPE.get(frame_type, 99999))),
                "ocr_preprocess_upscale": float(it.get("ocr_preprocess_upscale", 1.0)),
            })

    print(f"3) Parsing kept keyframes with YOLO+OCR concurrently... to_parse={len(to_parse)}")

    if to_parse:
        yolo_jobs = sum(1 for j in to_parse if j.get("parse_mode") == "yolo_ocr")
        ocr_only_jobs = len(to_parse) - yolo_jobs
        enable_yolo = yolo_jobs > 0

        print("    [step3] starting ProcessPoolExecutor...")
        print(f"    [step3] PARSE_WORKERS={PARSE_WORKERS} (each worker loads YOLO + OCR backend once)")
        print(
            f"    [step3] YOLO_DEVICE={YOLO_DEVICE} YOLO_IMGSZ={YOLO_IMGSZ} | "
            f"OCR_MODE={OCR_MODE} OCR_BACKEND={OCR_BACKEND} OCR_GPU={USE_GPU} angle_cls=False"
        )
        print(f"    [step3] OCR global upscale={OCR_GLOBAL_UPSCALE}")
        print(f"    [step3] OCR crops: max_regions={OCR_CROP_MAX_REGIONS} scale_by_type={OCR_CROP_SCALE_BY_TYPE}")
        print(f"    [step3] OCR frame upscale by type={OCR_FRAME_UPSCALE_BY_TYPE}")
        print(f"    [step3] OCR save max_w by type={OCR_SAVE_MAX_W_BY_TYPE}")
        print(f"    [step3] Parse resize max_w_by_type={PARSE_MAX_W_BY_TYPE}")
        print(f"    [step3] parse_mode split: yolo_ocr={yolo_jobs}, ocr_only={ocr_only_jobs}")

        t0 = time.perf_counter()
        err_count = 0
        worker_count = max(1, int(PARSE_WORKERS))

        if worker_count == 1:
            print("    [step3] using single-process parser (CUDA-safe path)")
            _worker_init(
                str(LAYOUT_YOLO_WEIGHTS),
                str(OCR_LANG),
                bool(enable_yolo),
                str(OCR_BACKEND),
                bool(USE_GPU),
            )
            for done_count, job in enumerate(to_parse, start=1):
                res = _parse_one_keyframe(job)
                kidx = int(res.get("keyframe_idx", -1))
                if kidx < 0 or kidx >= len(all_selected):
                    continue

                if "error" in res:
                    err_count += 1
                    all_selected[kidx]["screen_parse_error"] = res["error"]
                else:
                    all_selected[kidx]["on_screen_text"] = res.get("on_screen_text", [])[:MAX_OCR_LINES]
                    all_selected[kidx]["screen_parse"] = res.get("screen_parse")
                    tm = all_selected[kidx].get("timings_ms", {}) or {}
                    tm.update(res.get("parse_timings_ms", {}) or {})
                    all_selected[kidx]["timings_ms"] = tm

                if done_count == 1 or done_count % 2 == 0 or done_count == len(to_parse):
                    print(f"    [step3] progress {done_count}/{len(to_parse)} parsed (errors={err_count})")
        else:
            # CUDA + fork can hang after CLIP initializes GPU in parent process.
            # Use spawn to isolate worker CUDA contexts safely.
            mp_ctx = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp_ctx,
                initializer=_worker_init,
                initargs=(
                    str(LAYOUT_YOLO_WEIGHTS),
                    str(OCR_LANG),
                    bool(enable_yolo),
                    str(OCR_BACKEND),
                    bool(USE_GPU),
                ),
            ) as ex:
                futs = [ex.submit(_parse_one_keyframe, job) for job in to_parse]

                done_count = 0
                t_last_report = time.perf_counter()

                for fut in cf.as_completed(futs):
                    res = fut.result()
                    kidx = int(res.get("keyframe_idx", -1))
                    if kidx < 0 or kidx >= len(all_selected):
                        continue

                    done_count += 1

                    if "error" in res:
                        err_count += 1
                        all_selected[kidx]["screen_parse_error"] = res["error"]
                    else:
                        all_selected[kidx]["on_screen_text"] = res.get("on_screen_text", [])[:MAX_OCR_LINES]
                        all_selected[kidx]["screen_parse"] = res.get("screen_parse")
                        tm = all_selected[kidx].get("timings_ms", {}) or {}
                        tm.update(res.get("parse_timings_ms", {}) or {})
                        all_selected[kidx]["timings_ms"] = tm

                    now = time.perf_counter()
                    if (now - t_last_report) >= 1.0 or done_count == len(futs):
                        print(f"    [step3] progress {done_count}/{len(futs)} parsed (errors={err_count})")
                        t_last_report = now

        t3_ms = (time.perf_counter() - t0) * 1000.0
        timing_totals["parse_concurrent_ms"] += t3_ms
        print(f"    [step3] done in {t3_ms/1000.0:.2f}s (errors={err_count})")

    # Rebuild buckets from final frame_type
    buckets: Dict[str, List[dict]] = {k: [] for k in out_paths.keys()}
    for it in all_selected:
        ft = it.get("frame_type", "none")
        if ft not in buckets:
            ft = "none"
            it["frame_type"] = "none"
        buckets[ft].append(it)

    # Final writes
    t0 = time.perf_counter()
    safe_write_json(enriched_json, all_selected)
    for ft, p in out_paths.items():
        safe_write_json(p, buckets[ft])
    timing_totals["json_write_ms"] += (time.perf_counter() - t0) * 1000.0

    total_ms = (time.perf_counter() - t_total0) * 1000.0

    timing_summary = {
        "timing_totals_ms": {k: float(v) for k, v in timing_totals.items()},
        "total_ms": float(total_ms),
        "candidates": int(len(candidates)),
        "selected_frames": int(len(all_selected)),
        "parsed_frames": int(sum(1 for x in all_selected if isinstance(x.get("screen_parse"), dict))),
        "parse_workers": int(PARSE_WORKERS),
        "min_keyframe_gap_sec": float(MIN_KEYFRAME_GAP_SEC),
        "yolo_device": str(YOLO_DEVICE),
        "yolo_imgsz": int(YOLO_IMGSZ),
        "ocr_mode": str(OCR_MODE),
        "ocr_backend": str(OCR_BACKEND),
        "ocr_use_gpu": bool(USE_GPU),
        "ocr_angle_cls": False,
        "ocr_crop_max_regions": int(OCR_CROP_MAX_REGIONS),
        "ocr_crop_scale_by_type": dict(OCR_CROP_SCALE_BY_TYPE),
        "ocr_global_upscale": float(OCR_GLOBAL_UPSCALE),
        "ocr_frame_upscale_by_type": dict(OCR_FRAME_UPSCALE_BY_TYPE),
        "ocr_save_max_w_by_type": dict(OCR_SAVE_MAX_W_BY_TYPE),
        "parse_max_w_by_type": dict(PARSE_MAX_W_BY_TYPE),
        "keyframe_image_format": str(KEYFRAME_IMAGE_FORMAT),
        "keyframe_png_compression": int(KEYFRAME_PNG_COMPRESSION),
        "keyframe_jpeg_quality": int(KEYFRAME_JPEG_QUALITY),
    }
    safe_write_json(timing_json, timing_summary)

    print("\nDone.")
    print("Selected frames:", len(all_selected))
    print("Frames folder:", frames_dir)
    print("Parsed JSON:", enriched_json)
    print("Timing JSON:", timing_json)
    for ft, p in out_paths.items():
        print(ft, "->", p)

    print("\nTiming summary (ms):")
    for k, v in timing_totals.items():
        print(f"  {k}: {v:.0f}")
    print(f"  total_ms: {total_ms:.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise
