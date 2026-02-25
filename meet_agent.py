import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_core.output_parsers import PydanticOutputParser
except Exception:
    PydanticOutputParser = None  
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from PIL import Image
import imagehash

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    _SCENEDETECT_OK = True
except Exception:
    _SCENEDETECT_OK = False


class FrameAnalysis(BaseModel):
    view: str = Field(description="One of: slides, camera, mixed, other")
    likely_topic: str = Field(description="Short topic label (3-8 words)")
    visual_summary: str = Field(
        description="What is happening / what is shown on screen"
    )
    on_screen_text: List[str] = Field(
        description="Key readable text (slide titles, bullets, numbers)"
    )
    key_entities: List[str] = Field(
        description="People/product names/tools/keywords visible or strongly implied"
    )
    visible_name_labels: List[str] = Field(
        default_factory=list,
        description="Any participant names visible in UI tiles (if present)",
    )


class EvidenceRef(BaseModel):
    moment_id: int
    keyframe_idx: int
    start_sec: float
    end_sec: float


class DecisionItem(BaseModel):
    text: str
    refs: List[EvidenceRef] = Field(default_factory=list)


class ActionItem(BaseModel):
    text: str
    owner: Optional[str] = None
    refs: List[EvidenceRef] = Field(default_factory=list)


class ChapterSummary(BaseModel):
    title: str
    summary: str
    key_points: List[str]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]


class SpeakerLabelUpdate(BaseModel):
    speaker_id: int
    proposed_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class MeetingState(BaseModel):
    # speaker_id(str) -> {label, name, evidence[], candidates[]}
    speaker_map: Dict[str, dict] = Field(default_factory=dict)

    # perceptual hash -> {first_seen_sec, analysis, path}
    keyframe_memory: Dict[str, dict] = Field(default_factory=dict)

    running_summary: str = ""

    chapter_summaries: List[dict] = Field(default_factory=list)


@dataclass
class Keyframe:
    idx: int
    t_sec: float
    path: Path
    diff_score: float


@dataclass
class Utterance:
    start: float
    end: float
    speaker: Optional[int]
    text: str



EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s]{8,}\d)\b")

INTRO_PATTERNS = [
    re.compile(r"\b(i am|i'm|this is)\s+([A-Z][a-zA-Z]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bmy name is\s+([A-Z][a-zA-Z]{1,30})\b", re.IGNORECASE),
    re.compile(r"\b([A-Z][a-zA-Z]{1,30})\s+here\b", re.IGNORECASE),
]


def redact(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


def load_state(state_path: Path) -> MeetingState:
    if state_path.exists():
        return MeetingState.model_validate_json(state_path.read_text(encoding="utf-8"))
    return MeetingState()


def save_state(state_path: Path, state: MeetingState) -> None:
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def ensure_speaker(state: MeetingState, speaker_id: Optional[int]) -> None:
    if speaker_id is None:
        return
    sid = str(speaker_id)
    if sid not in state.speaker_map:
        state.speaker_map[sid] = {
            "label": f"Speaker {speaker_id}",
            "name": None,
            "evidence": [],
            "candidates": [],
        }


def get_speaker_label(state: MeetingState, speaker_id: Optional[int]) -> str:
    if speaker_id is None:
        return "Unknown"
    ensure_speaker(state, speaker_id)
    return state.speaker_map[str(speaker_id)]["label"]


def extract_self_intro_name(text: str) -> Optional[str]:
    t = text.strip()
    for pat in INTRO_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 2:
            name = m.group(2)
        else:
            name = m.group(1)
        if name:
            name = name.strip()
            return name[0].upper() + name[1:]
    return None


def maybe_update_speaker_name_from_self_intro(
    state: MeetingState, speaker_id: Optional[int], utter_text: str
) -> None:
    if speaker_id is None:
        return
    ensure_speaker(state, speaker_id)
    sid = str(speaker_id)
    if state.speaker_map[sid].get("name"):
        return

    name = extract_self_intro_name(utter_text)
    if name:
        state.speaker_map[sid]["name"] = name
        state.speaker_map[sid]["label"] = name
        state.speaker_map[sid]["evidence"].append(
            {"type": "self_intro", "text": redact(utter_text)[:220]}
        )


def add_name_candidates(
    state: MeetingState, speaker_id: Optional[int], candidates: List[str]
) -> None:
    if speaker_id is None or not candidates:
        return
    ensure_speaker(state, speaker_id)
    sid = str(speaker_id)
    existing = set(state.speaker_map[sid].get("candidates", []))
    for c in candidates:
        c = redact(c.strip())
        if c and c not in existing:
            state.speaker_map[sid]["candidates"].append(c)



def run_ffmpeg_extract_audio(video_path: Path, wav_out: Path) -> None:
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed.\n")
        print("Command:", " ".join(cmd))
        print("\nstdout:\n", e.stdout)
        print("\nstderr:\n", e.stderr)
        raise


def mad_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def extract_keyframes_by_change_detection(
    video_path: Path,
    out_dir: Path,
    sample_fps: float = 1.0,
    diff_threshold: float = 12.0,
    min_gap_sec: float = 3.0,
    max_frames: int = 80,
    resize_w: int = 320,
) -> List[Keyframe]:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    kept: List[Keyframe] = []
    last_kept_small = None
    last_kept_t = -1e9

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total_frames, desc="Scanning video (diff)", unit="frame")

    frame_idx = 0
    kept_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        t_sec = frame_idx / fps

        h, w = frame.shape[:2]
        new_w = resize_w
        new_h = int(h * (new_w / w))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        diff = 999.0 if last_kept_small is None else mad_diff(gray, last_kept_small)
        enough_time = (t_sec - last_kept_t) >= min_gap_sec
        enough_change = diff >= diff_threshold

        if (len(kept) == 0) or (enough_time and enough_change):
            out_path = frames_dir / f"frame_{kept_idx:04d}_{t_sec:.2f}s.jpg"
            cv2.imwrite(str(out_path), frame)
            kept.append(
                Keyframe(idx=kept_idx, t_sec=t_sec, path=out_path, diff_score=diff)
            )
            last_kept_small = gray
            last_kept_t = t_sec
            kept_idx += 1
            if len(kept) >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return kept


def _probe_video(video_path: Path) -> Tuple[float, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = float(frames / fps) if frames else 0.0
    cap.release()
    return fps, duration, frames


def _grab_frame_at_sec(video_path: Path, t_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def extract_keyframes_with_scenedetect(
    video_path: Path,
    out_dir: Path,
    threshold: float = 27.0,
    min_scene_len_sec: float = 2.0,
    max_frames: int = 80,
) -> List[Keyframe]:
    """
    Preferred: use PySceneDetect to get robust scene boundaries (better for slide changes).
    We save one representative frame per detected scene.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    fps, duration, _ = _probe_video(video_path)
    if duration <= 0:
        raise RuntimeError("Could not determine video duration for scene detection.")

    if not _SCENEDETECT_OK:
        raise RuntimeError("PySceneDetect not available.")

    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=int(max(1, round(min_scene_len_sec * fps))),
        )
    )
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()

    if not scenes:
        scenes = [(video.base_timecode, video.base_timecode + int(duration * fps))]

    keyframes: List[Keyframe] = []
    kept_idx = 0

    for start_tc, end_tc in scenes:
        start_sec = float(start_tc.get_seconds())
        end_sec = float(end_tc.get_seconds())
        if end_sec <= start_sec:
            continue

        rep_sec = min(end_sec - 0.05, start_sec + 0.20)
        rep_sec = max(start_sec, rep_sec)

        frame = _grab_frame_at_sec(video_path, rep_sec)
        if frame is None:
            continue

        out_path = frames_dir / f"frame_{kept_idx:04d}_{start_sec:.2f}s.jpg"
        cv2.imwrite(str(out_path), frame)
        keyframes.append(
            Keyframe(idx=kept_idx, t_sec=start_sec, path=out_path, diff_score=-1.0)
        )
        kept_idx += 1

        if len(keyframes) >= max_frames:
            break

    keyframes.sort(key=lambda k: k.t_sec)
    return keyframes


def deepgram_transcribe_utterances(wav_path: Path, api_key: str) -> List[Utterance]:
    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-3",
        "smart_format": "true",
        "punctuate": "true",
        "diarize": "true",
        "utterances": "true",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    audio_bytes = wav_path.read_bytes()
    resp = requests.post(
        url, params=params, headers=headers, data=audio_bytes, timeout=600
    )
    resp.raise_for_status()
    data = resp.json()

    utts: List[Utterance] = []
    for u in data.get("results", {}).get("utterances", []):
        utts.append(
            Utterance(
                start=float(u.get("start", 0.0)),
                end=float(u.get("end", 0.0)),
                speaker=u.get("speaker", None),
                text=(u.get("transcript") or "").strip(),
            )
        )

    if not utts:
        alt = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
        )
        txt = (alt.get("transcript") or "").strip()
        if txt:
            utts = [Utterance(start=0.0, end=0.0, speaker=None, text=txt)]

    utts.sort(key=lambda x: (x.start, x.end))
    return utts



def phash_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    return str(imagehash.phash(img))


def _redact_frame_analysis(fa: FrameAnalysis) -> FrameAnalysis:
    fa.on_screen_text = [redact(x) for x in fa.on_screen_text]
    fa.visual_summary = redact(fa.visual_summary)
    fa.likely_topic = redact(fa.likely_topic)
    fa.key_entities = [redact(x) for x in fa.key_entities]
    fa.visible_name_labels = [redact(x) for x in fa.visible_name_labels]
    return fa


def gemini_analyze_frame(client: genai.Client, image_path: Path) -> FrameAnalysis:
    image_bytes = image_path.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    prompt = """
Analyze a single frame from a recorded online meeting and output JSON.

Rules:
- view: slides if mostly screen share/slide/doc/app; camera if mostly participants; mixed if both; other otherwise.
- on_screen_text: capture the most important visible text (titles, headers, bullets, numbers). Keep it short (max ~12 lines).
- visible_name_labels: list any participant names visible in the UI tiles (if present). If none, [].
- visual_summary: describe what is shown and what is happening, in 2-4 sentences.
- likely_topic: concise topic label (3-8 words).
- key_entities: relevant names/tools/keywords.
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image_part, prompt],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": FrameAnalysis.model_json_schema(),
            "temperature": 0.2,
        },
    )
    fa = FrameAnalysis.model_validate_json(resp.text)
    return _redact_frame_analysis(fa)


def get_embedding(client: genai.Client, text: str) -> np.ndarray:
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    vec = result.embeddings[0].values
    return np.array(vec, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)



def build_text_llm(
    api_key: str, model_name: str, temperature: float = 0.2
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )


def summarize_chapter_langchain(
    llm: ChatGoogleGenerativeAI,
    state: MeetingState,
    chapter_moments: List[dict],
) -> ChapterSummary:
    """
    Chapter summary with grounded decisions/action items.
    Ensures we return a ChapterSummary object even if parser returns dict.
    """
    if PydanticOutputParser is not None:
        parser = PydanticOutputParser(pydantic_object=ChapterSummary)
    else:
        parser = JsonOutputParser(pydantic_object=ChapterSummary)

    prior_titles = [c.get("title", "") for c in state.chapter_summaries][-8:]
    speakers_brief = {
        sid: {"label": v.get("label"), "evidence": v.get("evidence", [])[:2]}
        for sid, v in state.speaker_map.items()
    }

    catalog = []
    for m in chapter_moments:
        frame = m.get("frame") or {}
        is_rep = bool(m.get("is_repeat_frame", False))
        repeat_note = m.get("repeat_note", "")

        screen_lines = [] if is_rep else (frame.get("on_screen_text") or [])[:6]
        visual = repeat_note if is_rep else (frame.get("visual_summary") or "")

        catalog.append(
            {
                "moment_id": m["moment_id"],
                "t": f"{m['start_sec']:.1f}-{m['end_sec']:.1f}",
                "keyframe_idx": m["keyframe_idx"],
                "topic": frame.get("likely_topic", ""),
                "view": frame.get("view", ""),
                "screen": [redact(x) for x in screen_lines],
                "visual": redact(visual),
                "speech": redact(m.get("speech_snippet", "")[:700]),
                "repeat_frame": is_rep,
                "repeat_first_seen_sec": m.get("repeat_first_seen_sec", None),
            }
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a meeting summarizer. Output valid JSON only.\n"
                "Be faithful to the provided content.\n"
                "If something is not present, return empty lists.\n"
                "Do not invent names, numbers, or actions.\n"
                "Ground decisions and action items with refs from the moment catalog.\n",
            ),
            (
                "human",
                "Context so far (running summary):\n{running_summary}\n\n"
                "Known speakers:\n{speakers}\n\n"
                "Previous chapter titles:\n{prior_titles}\n\n"
                "Now summarize THIS chapter using the moment catalog.\n\n"
                "Output JSON with:\n"
                "- title\n"
                "- summary (4-6 lines)\n"
                "- key_points (5-10 strings)\n"
                "- decisions: list of objects {{text, refs:[{{moment_id,keyframe_idx,start_sec,end_sec}}...]}}\n"
                "- action_items: list of objects {{text, owner(optional), refs:[{{moment_id,keyframe_idx,start_sec,end_sec}}...]}}\n\n"
                "Rules for refs:\n"
                "- For each decision/action item, include 1-3 refs that support it.\n"
                "- refs must use ONLY moment_id/keyframe_idx/time ranges that exist in the catalog.\n"
                "- If you cannot confidently ground it, leave refs as [].\n\n"
                "{format_instructions}\n\n"
                "Moment catalog:\n{moments_json}",
            ),
        ]
    )

    chain = prompt | llm | parser
    out = chain.invoke(
        {
            "running_summary": state.running_summary[-2000:],
            "speakers": json.dumps(speakers_brief, ensure_ascii=False, indent=2),
            "prior_titles": json.dumps(prior_titles, ensure_ascii=False),
            "format_instructions": parser.get_format_instructions(),
            "moments_json": json.dumps(catalog, ensure_ascii=False),
        }
    )

   
    if isinstance(out, ChapterSummary):
        return out
    return ChapterSummary.model_validate(out)


def update_running_summary(
    llm: ChatGoogleGenerativeAI, state: MeetingState, new_chapter: ChapterSummary
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You maintain a running meeting summary. Keep it compact but complete.",
            ),
            (
                "human",
                "Current running summary:\n{old}\n\n"
                "New chapter summary (JSON):\n{new}\n\n"
                "Update the running summary.\n"
                "Rules:\n"
                "- Keep under ~20 lines\n"
                "- Include decisions + action items if any\n"
                "- Use consistent speaker labels from the speaker map\n"
                "Return plain text only.",
            ),
        ]
    )
    chain = prompt | llm
    res = chain.invoke(
        {
            "old": state.running_summary.strip(),
            "new": json.dumps(new_chapter.model_dump(), ensure_ascii=False, indent=2),
        }
    )
    return res.content.strip()


def final_text_only_summary(llm: ChatGoogleGenerativeAI, state: MeetingState) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write a final meeting summary in plain text. No markdown. No JSON.",
            ),
            (
                "human",
                "Speaker map:\n{speakers}\n\n"
                "Chapter summaries (with grounded decisions/action items):\n{chapters}\n\n"
                "Running summary:\n{running}\n\n"
                "Write a text-only final summary of the entire meeting.\n"
                "Include:\n"
                "1) What was discussed (structured)\n"
                "2) Decisions (with brief evidence pointers like m12/k3)\n"
                "3) Action items (with owners if known; include evidence pointers)\n"
                "4) Open questions / follow-ups\n"
                "Return plain text only.",
            ),
        ]
    )
    chain = prompt | llm

    chapters_txt = []
    for i, c in enumerate(state.chapter_summaries, start=1):
        chapters_txt.append(f"Chapter {i}: {c.get('title','')}\n{c.get('summary','')}")
        if c.get("decisions"):
            chapters_txt.append("Decisions:")
            for d in c["decisions"]:
                refs = d.get("refs", [])
                ref_str = (
                    ", ".join([f"m{r['moment_id']}/k{r['keyframe_idx']}" for r in refs])
                    if refs
                    else ""
                )
                chapters_txt.append(
                    f"- {d.get('text','')}" + (f" ({ref_str})" if ref_str else "")
                )
        if c.get("action_items"):
            chapters_txt.append("Action items:")
            for a in c["action_items"]:
                refs = a.get("refs", [])
                ref_str = (
                    ", ".join([f"m{r['moment_id']}/k{r['keyframe_idx']}" for r in refs])
                    if refs
                    else ""
                )
                owner = a.get("owner")
                owner_str = f"[{owner}] " if owner else ""
                chapters_txt.append(
                    f"- {owner_str}{a.get('text','')}"
                    + (f" ({ref_str})" if ref_str else "")
                )
        chapters_txt.append("")

    res = chain.invoke(
        {
            "speakers": json.dumps(state.speaker_map, ensure_ascii=False, indent=2),
            "chapters": "\n".join(chapters_txt),
            "running": state.running_summary.strip(),
        }
    )
    return res.content.strip()




def collect_utterances_in_window(
    utterances: List[Utterance], start: float, end: float
) -> List[Utterance]:
    out: List[Utterance] = []
    for u in utterances:
        if u.end and u.end < start:
            continue
        if u.start > end:
            break
        if (u.start <= end) and (u.end >= start):
            out.append(u)
    return out


def make_boundaries(
    keyframes: List[Keyframe],
    utterances: List[Utterance],
    video_duration_sec: float,
    min_boundary_gap: float = 0.6,
) -> List[float]:
    """
    Boundaries react to BOTH:
    - visual changes (keyframe timestamps)
    - speech pauses / turn boundaries (utterance end timestamps)

    Merge -> sort -> de-duplicate.
    """
    b: List[float] = [0.0]

    for k in keyframes:
        b.append(float(k.t_sec))

    for u in utterances:
        if u.end and u.end > 0:
            b.append(float(u.end))

    b.append(float(video_duration_sec))
    b.sort()

    dedup: List[float] = []
    for x in b:
        if not dedup or abs(x - dedup[-1]) >= min_boundary_gap:
            dedup.append(x)
    return dedup


def attach_keyframe_to_time(keyframes: List[Keyframe], t_sec: float) -> Keyframe:
    if not keyframes:
        raise RuntimeError("No keyframes available.")
    best = keyframes[0]
    for k in keyframes:
        if k.t_sec <= t_sec:
            best = k
        else:
            break
    return best



def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to meeting.mp4")
    ap.add_argument("--out", required=True, help="Output folder")

    ap.add_argument(
        "--scene_threshold",
        type=float,
        default=27.0,
        help="PySceneDetect ContentDetector threshold",
    )
    ap.add_argument(
        "--scene_min_len_sec",
        type=float,
        default=2.0,
        help="Minimum scene length in seconds",
    )

    ap.add_argument("--sample_fps", type=float, default=1.0)
    ap.add_argument("--diff_threshold", type=float, default=12.0)
    ap.add_argument("--min_gap_sec", type=float, default=3.0)
    ap.add_argument("--max_frames", type=int, default=80)

    ap.add_argument(
        "--min_boundary_gap",
        type=float,
        default=0.6,
        help="Dedup close boundaries (sec)",
    )

    ap.add_argument("--chapter_min_sec", type=float, default=300.0)
    ap.add_argument("--chapter_max_sec", type=float, default=900.0)
    ap.add_argument("--chapter_sim_threshold", type=float, default=0.72)

    ap.add_argument(
        "--text_model", type=str, default=os.getenv("TEXT_MODEL", "gemini-2.5-flash")
    )

    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "state.json"
    state = load_state(state_path)

    gemini_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it in your environment."
        )
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not deepgram_key:
        raise RuntimeError("Missing DEEPGRAM_API_KEY. Set it in your environment.")

    vision_client = genai.Client(api_key=gemini_key)
    text_llm = build_text_llm(
        api_key=gemini_key, model_name=args.text_model, temperature=0.2
    )

    fps, video_duration_sec, _ = _probe_video(video_path)

    # 1) Extract audio
    wav_path = out_dir / "audio_16k_mono.wav"
    if not wav_path.exists():
        print("1) Extracting audio with ffmpeg...")
        run_ffmpeg_extract_audio(video_path, wav_path)
    else:
        print("1) Audio exists, skipping.")

    # 2) Speech-to-text (Deepgram)
    utterances_path = out_dir / "utterances.json"
    if utterances_path.exists():
        utterances = [
            Utterance(**u)
            for u in json.loads(utterances_path.read_text(encoding="utf-8"))
        ]
        print("2) Loaded cached utterances.json")
    else:
        print("2) Transcribing audio with Deepgram...")
        utterances = deepgram_transcribe_utterances(wav_path, deepgram_key)
        utterances_path.write_text(
            json.dumps([u.__dict__ for u in utterances], indent=2), encoding="utf-8"
        )

    for u in utterances:
        ensure_speaker(state, u.speaker)
        maybe_update_speaker_name_from_self_intro(state, u.speaker, u.text)
    save_state(state_path, state)

    transcript_lines = []
    for u in utterances:
        label = get_speaker_label(state, u.speaker)
        transcript_lines.append(
            f"[{u.start:8.2f}-{u.end:8.2f}] {label}: {redact(u.text)}"
        )
    (out_dir / "transcript.txt").write_text(
        "\n".join(transcript_lines), encoding="utf-8"
    )

    # 3) Keyframes 
    keyframes_path = out_dir / "keyframes.json"
    if keyframes_path.exists():
        raw = json.loads(keyframes_path.read_text(encoding="utf-8"))
        keyframes = [
            Keyframe(
                idx=k["idx"],
                t_sec=k["t_sec"],
                path=Path(k["path"]),
                diff_score=k.get("diff_score", -1.0),
            )
            for k in raw
        ]
        keyframes.sort(key=lambda k: k.t_sec)
        print("3) Loaded cached keyframes.json")
    else:
        print("3) Extracting keyframes with PySceneDetect (preferred)...")
        try:
            keyframes = extract_keyframes_with_scenedetect(
                video_path=video_path,
                out_dir=out_dir,
                threshold=args.scene_threshold,
                min_scene_len_sec=args.scene_min_len_sec,
                max_frames=args.max_frames,
            )
            print(f"   SceneDetect keyframes: {len(keyframes)}")
        except Exception as e:
            print(
                f"   SceneDetect failed ({e}). Falling back to diff-based keyframes..."
            )
            keyframes = extract_keyframes_by_change_detection(
                video_path=video_path,
                out_dir=out_dir,
                sample_fps=args.sample_fps,
                diff_threshold=args.diff_threshold,
                min_gap_sec=args.min_gap_sec,
                max_frames=args.max_frames,
            )
            print(f"   Diff keyframes: {len(keyframes)}")

        keyframes_path.write_text(
            json.dumps(
                [
                    {
                        "idx": k.idx,
                        "t_sec": k.t_sec,
                        "path": str(k.path),
                        "diff_score": k.diff_score,
                    }
                    for k in keyframes
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    # 4) Gemini vision analysis per keyframe + keyframe memory (pHash)
    frame_analysis_path = out_dir / "frame_analysis.json"
    frame_analysis: Dict[str, dict] = {}

    if frame_analysis_path.exists():
        frame_analysis = json.loads(frame_analysis_path.read_text(encoding="utf-8"))
        print("4) Loaded cached frame_analysis.json")
    else:
        print("4) Analyzing keyframes with Gemini vision (with repeat-frame memory)...")
        for k in tqdm(keyframes, desc="Gemini vision", unit="frame"):
            ph = phash_image(k.path)

            if ph in state.keyframe_memory:
                frame_analysis[str(k.idx)] = state.keyframe_memory[ph]["analysis"]
            else:
                fa = gemini_analyze_frame(vision_client, k.path)
                frame_analysis[str(k.idx)] = fa.model_dump()
                state.keyframe_memory[ph] = {
                    "first_seen_sec": float(k.t_sec),
                    "analysis": fa.model_dump(),
                    "path": str(k.path),
                }
                save_state(state_path, state)

        frame_analysis_path.write_text(
            json.dumps(frame_analysis, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        save_state(state_path, state)

    # 5) Build moments using merged boundaries 
    print("5) Building moments (boundaries = keyframes + utterance ends)...")
    boundaries = make_boundaries(
        keyframes=keyframes,
        utterances=utterances,
        video_duration_sec=(
            video_duration_sec
            if video_duration_sec > 0
            else (keyframes[-1].t_sec + 10.0)
        ),
        min_boundary_gap=args.min_boundary_gap,
    )


    phash_cache: Dict[int, str] = {}
    repeat_cache: Dict[int, bool] = {}
    first_seen_cache: Dict[int, float] = {}

    seen_ph = set()
    for k in keyframes:
        ph = phash_image(k.path)
        phash_cache[k.idx] = ph
        repeat_cache[k.idx] = ph in seen_ph
        if ph in state.keyframe_memory:
            first_seen_cache[k.idx] = float(
                state.keyframe_memory[ph].get("first_seen_sec", k.t_sec)
            )
        else:
            first_seen_cache[k.idx] = float(k.t_sec)
        seen_ph.add(ph)

    moments: List[dict] = []
    moment_id = 0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue

        kf = attach_keyframe_to_time(keyframes, start)
        fa = frame_analysis.get(str(kf.idx), {})
        visible_labels = (
            fa.get("visible_name_labels", []) if isinstance(fa, dict) else []
        )

        speech = collect_utterances_in_window(utterances, start, end)

        for u in speech:
            add_name_candidates(state, u.speaker, visible_labels)
            maybe_update_speaker_name_from_self_intro(state, u.speaker, u.text)

        lines = []
        for u in speech:
            if not u.text:
                continue
            label = get_speaker_label(state, u.speaker)
            lines.append(f"{label}: {redact(u.text)}")
        speech_text = " ".join(lines).strip()

        is_repeat = bool(repeat_cache.get(kf.idx, False))
        first_seen = float(first_seen_cache.get(kf.idx, kf.t_sec))

        repeat_note = ""
        if is_repeat:
            repeat_note = f"Same slide as earlier (first seen at {first_seen:.1f}s); visual unchanged; focus on new speech."

        moments.append(
            {
                "moment_id": moment_id,
                "start_sec": float(start),
                "end_sec": float(end),
                "keyframe_path": str(kf.path),
                "keyframe_idx": int(kf.idx),
                "is_repeat_frame": is_repeat,
                "repeat_first_seen_sec": first_seen if is_repeat else None,
                "repeat_note": repeat_note,
                "frame": fa,
                "speech_snippet": speech_text,
            }
        )
        moment_id += 1

    save_state(state_path, state)
    (out_dir / "moments.json").write_text(
        json.dumps(moments, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6) Chapter segmentation based on semantic similarity of consecutive moments
    print("6) Splitting into chapters (context-driven)...")
    moment_embs: List[np.ndarray] = []

    for m in tqdm(moments, desc="Embedding moments", unit="moment"):
        frame = m.get("frame") or {}

        if m.get("is_repeat_frame"):
            screen = ""
            visual = m.get("repeat_note", "")
        else:
            screen = " | ".join((frame.get("on_screen_text") or [])[:8])
            visual = frame.get("visual_summary", "")

        text = (
            f"Topic: {frame.get('likely_topic','')}\n"
            f"View: {frame.get('view','')}\n"
            f"Screen: {redact(screen)}\n"
            f"Visual: {redact(visual)}\n"
            f"Speech: {redact(m.get('speech_snippet','')[:900])}\n"
            f"RepeatFrame: {m.get('is_repeat_frame', False)}"
        )
        moment_embs.append(get_embedding(vision_client, text))

    chapters: List[List[dict]] = []
    cur: List[dict] = []
    cur_start = moments[0]["start_sec"] if moments else 0.0

    for idx in range(len(moments)):
        cur.append(moments[idx])
        dur = cur[-1]["end_sec"] - cur_start

        split = False
        if len(cur) >= 2:
            sim = cosine(moment_embs[idx - 1], moment_embs[idx])
            if (sim < args.chapter_sim_threshold) and (dur >= args.chapter_min_sec):
                split = True
        if dur >= args.chapter_max_sec:
            split = True

        if split:
            chapters.append(cur)
            cur = []
            if idx + 1 < len(moments):
                cur_start = moments[idx + 1]["start_sec"]

    if cur:
        chapters.append(cur)

    # 7) Chapter summaries with grounded evidence refs
    print("7) Summarizing chapters (grounded decisions/action items)...")
    chapter_objs: List[dict] = []
    state.chapter_summaries = state.chapter_summaries or []

    for ch in tqdm(chapters, desc="Chapter summary", unit="chapter"):
        start = ch[0]["start_sec"]
        end = ch[-1]["end_sec"]

        ch_sum = summarize_chapter_langchain(text_llm, state, ch)

        
        state.chapter_summaries.append(
            {
                "start_sec": start,
                "end_sec": end,
                "title": ch_sum.title,
                "summary": ch_sum.summary,
                "decisions": [d.model_dump() for d in ch_sum.decisions],
                "action_items": [a.model_dump() for a in ch_sum.action_items],
            }
        )

        state.running_summary = update_running_summary(text_llm, state, ch_sum)
        save_state(state_path, state)

        chapter_objs.append(
            {
                "start_sec": start,
                "end_sec": end,
                "summary": ch_sum.model_dump(),
                "moments": ch,
            }
        )

    (out_dir / "chapters.json").write_text(
        json.dumps(chapter_objs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 8) Markdown report 
    report = []
    report.append(f"# Meeting report: {video_path.name}\n")
    report.append("## Speaker map\n")
    for sid, info in sorted(state.speaker_map.items(), key=lambda x: int(x[0])):
        report.append(f"- {sid}: {info.get('label')}\n")
    report.append("\n")

    for ci, ch in enumerate(chapter_objs, start=1):
        s = ch["summary"]
        report.append(
            f"## Chapter {ci}: {s['title']} ({ch['start_sec']:.1f}s–{ch['end_sec']:.1f}s)\n"
        )
        report.append(s["summary"] + "\n")

        if s.get("key_points"):
            report.append(
                "Key points:\n" + "\n".join([f"- {x}" for x in s["key_points"]]) + "\n"
            )

        if s.get("decisions"):
            report.append("Decisions (grounded):\n")
            for d in s["decisions"]:
                refs = d.get("refs", [])
                ref_str = (
                    ", ".join(
                        [
                            f"m{r['moment_id']}/k{r['keyframe_idx']}({r['start_sec']:.1f}-{r['end_sec']:.1f}s)"
                            for r in refs
                        ]
                    )
                    if refs
                    else ""
                )
                report.append(
                    f"- {d.get('text','')}"
                    + (f" [{ref_str}]" if ref_str else "")
                    + "\n"
                )
            report.append("\n")

        if s.get("action_items"):
            report.append("Action items (grounded):\n")
            for a in s["action_items"]:
                refs = a.get("refs", [])
                ref_str = (
                    ", ".join(
                        [
                            f"m{r['moment_id']}/k{r['keyframe_idx']}({r['start_sec']:.1f}-{r['end_sec']:.1f}s)"
                            for r in refs
                        ]
                    )
                    if refs
                    else ""
                )
                owner = a.get("owner")
                owner_str = f"{owner}: " if owner else ""
                report.append(
                    f"- {owner_str}{a.get('text','')}"
                    + (f" [{ref_str}]" if ref_str else "")
                    + "\n"
                )
            report.append("\n")

        report.append("Moments:\n")
        for m in ch["moments"]:
            img_rel = Path(m["keyframe_path"]).relative_to(out_dir)
            frame = m.get("frame") or {}
            repeat_note = ""
            if m.get("is_repeat_frame"):
                repeat_note = (
                    f" (repeat; first seen {m.get('repeat_first_seen_sec', 0.0):.1f}s)"
                )
            report.append(
                f"- m{m['moment_id']}: {m['start_sec']:.1f}s–{m['end_sec']:.1f}s | "
                f"k{m['keyframe_idx']} | {frame.get('likely_topic','')} | {frame.get('view','')}{repeat_note}\n"
            )
            report.append(f"  - Frame: ![]({img_rel.as_posix()})\n")

            if not m.get("is_repeat_frame"):
                ost = frame.get("on_screen_text") or []
                if ost:
                    report.append(
                        "  - On-screen text:\n"
                        + "\n".join([f"    - {redact(t)}" for t in ost[:10]])
                        + "\n"
                    )
                vs = frame.get("visual_summary", "")
                if vs:
                    report.append("  - Visual:\n    - " + redact(vs) + "\n")
            else:
                report.append(
                    "  - Visual delta:\n    - " + m.get("repeat_note", "") + "\n"
                )

            if m.get("speech_snippet"):
                report.append(
                    "  - Speech:\n    - "
                    + redact(m["speech_snippet"][:900]).replace("\n", " ")
                    + "\n"
                )
        report.append("\n")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    # 9) Final text-only summary of entire meeting
    meeting_summary = final_text_only_summary(text_llm, state)
    (out_dir / "meeting_summary.txt").write_text(meeting_summary, encoding="utf-8")

    print(f"\nDone. Outputs written to: {out_dir}")
    print(f"- {out_dir / 'report.md'}")
    print(f"- {out_dir / 'meeting_summary.txt'}")
    print(f"- {out_dir / 'transcript.txt'}")
    print(f"- {out_dir / 'state.json'}")


if __name__ == "__main__":
    main()