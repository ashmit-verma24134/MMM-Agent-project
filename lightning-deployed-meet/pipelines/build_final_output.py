# build_final_output.py
# Usage:
#   pip install google-genai pydantic python-dotenv
#   set GEMINI_API_KEY=...
#   python build_final_output.py ^
#       --keyframes "C:\meet-agent\out_folder\keyframes_with_utterances.json" ^
#       --out "C:\meet-agent\out_folder\final_output.json" ^
#       --model "gemini-2.5-flash"

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq

def groq_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment.")
    return Groq(api_key=api_key)
# -----------------------------
# Helpers
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def call_llm_structured(
    client: Groq,
    model: str,
    system_instruction: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> Dict[str, Any]:

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

            text_output = response.choices[0].message.content.strip()

            return json.loads(text_output)

        except Exception as e:
            last_err = e
            time.sleep(0.7 * attempt)

    raise RuntimeError(f"Groq structured call failed: {last_err}")
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sec_to_hhmmss(t: float) -> str:
    t = max(0.0, float(t))
    hh = int(t // 3600)
    mm = int((t % 3600) // 60)
    ss = int(t % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def safe_join_text(lines: List[str], max_chars: int = 8000) -> str:
    """Join lines but prevent prompt bloat."""
    out = []
    total = 0
    for ln in lines:
        if total + len(ln) + 1 > max_chars:
            break
        out.append(ln)
        total += len(ln) + 1
    return "\n".join(out)


def frame_signature(frame: Optional[Dict[str, Any]]) -> str:
    """Build a signature string for similarity comparison to previous keyframe."""
    if not frame:
        return ""
    on_screen = frame.get("on_screen_text") or []
    screen_parse = frame.get("screen_parse") or {}
    screen_parse_text = summarize_screen_parse(screen_parse, max_regions=3, max_region_lines=6, max_ocr_lines=30, max_chars=2500)
    on_screen_small = safe_join_text(on_screen[:80], max_chars=2500)
    return f"{on_screen_small}\n{screen_parse_text}"


def diff_lists(prev: List[str], cur: List[str], max_items: int = 25) -> Tuple[List[str], List[str]]:
    prev_set, cur_set = set(prev), set(cur)
    added = [x for x in cur if x not in prev_set][:max_items]
    removed = [x for x in prev if x not in cur_set][:max_items]
    return added, removed


def summarize_screen_parse(
    screen_parse: Optional[Dict[str, Any]],
    max_regions: int = 8,
    max_region_lines: int = 12,
    max_ocr_lines: int = 120,
    max_chars: int = 9000,
) -> str:
    if not isinstance(screen_parse, dict) or not screen_parse:
        return "unknown"

    parts: List[str] = []
    frame_w = screen_parse.get("frame_w")
    frame_h = screen_parse.get("frame_h")
    if frame_w is not None and frame_h is not None:
        parts.append(f"frame_size: {frame_w}x{frame_h}")

    regions = screen_parse.get("layout_regions") or []
    if regions:
        region_lines: List[str] = []
        for i, region in enumerate(regions[:max_regions]):
            label = region.get("label", "unknown")
            conf = region.get("conf", "unknown")
            box = region.get("box", [])
            text_lines = region.get("text_lines") or []
            text_lines_clean = [str(x).strip() for x in text_lines if str(x).strip()][:max_region_lines]
            text_preview = " | ".join(text_lines_clean)
            region_lines.append(
                f"region[{i}] label={label}, conf={conf}, box={box}, text_lines={text_preview}"
            )
        parts.append("layout_regions:\n" + "\n".join(region_lines))

    ocr_lines = screen_parse.get("ocr_lines") or []
    if ocr_lines:
        ocr_text: List[str] = []
        for item in ocr_lines[:max_ocr_lines]:
            txt = str(item.get("text", "")).strip()
            if txt:
                ocr_text.append(txt)
        if ocr_text:
            parts.append("ocr_lines:\n" + safe_join_text(ocr_text, max_chars=max_chars))

    merged = "\n\n".join(parts).strip()
    if not merged:
        return "unknown"
    return merged[:max_chars]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def build_content_change_summary(
    prev_content_summary: Optional[str],
    cur_content_summary: Optional[str],
    max_items: int = 6,
) -> str:
    prev = (prev_content_summary or "").strip()
    cur = (cur_content_summary or "").strip()
    if not prev:
        return "Initial keyframe in sequence; no previous content summary to diff against."
    if not cur:
        return "Current content summary is empty or unknown; unable to compute precise content diff."
    if prev == cur:
        return "No material content-summary change from the previous keyframe."

    prev_sentences = split_sentences(prev)
    cur_sentences = split_sentences(cur)
    prev_set = set(prev_sentences)
    cur_set = set(cur_sentences)

    added = [s for s in cur_sentences if s not in prev_set][:max_items]
    removed = [s for s in prev_sentences if s not in cur_set][:max_items]

    # If sentence-level diff fails (e.g., heavy rewrites), use token-level fallback.
    if not added and not removed:
        prev_tokens = set(tokenize(prev))
        cur_tokens = set(tokenize(cur))
        added_tokens = sorted(list(cur_tokens - prev_tokens))[:12]
        removed_tokens = sorted(list(prev_tokens - cur_tokens))[:12]
        if not added_tokens and not removed_tokens:
            return "Content summary wording changed but underlying content differences are unclear."
        out = []
        if added_tokens:
            out.append("Added/updated terms: " + ", ".join(added_tokens))
        if removed_tokens:
            out.append("Removed/de-emphasized terms: " + ", ".join(removed_tokens))
        return " ".join(out)

    chunks = []
    if added:
        chunks.append(
            "Added/updated in current content summary: "
            + " ; ".join(a[:240] for a in added)
        )
    if removed:
        chunks.append(
            "Removed/de-emphasized vs previous content summary: "
            + " ; ".join(r[:240] for r in removed)
        )
    return " ".join(chunks).strip()


def extract_speakers_from_utterances(utterances: List[Dict[str, Any]]) -> List[str]:
    """Unique speakers in order of first appearance."""
    seen = set()
    out = []
    for u in utterances or []:
        spk = str(u.get("speaker", "")).strip()
        if not spk:
            spk = "unknown"
        if spk not in seen:
            seen.add(spk)
            out.append(spk)
    return out


# -----------------------------
# Pydantic schema for Gemini
# -----------------------------
class FrameChange(BaseModel):
    changed_summary: str = Field(
        ...,
        description="Only the content-summary diff from previous keyframe to current keyframe.",
    )
    possible_reason: str = Field(
        ...,
        description="Why it could have happened (grounded in utterances/on-screen info; if unknown say unknown).",
    )
    added_elements: List[str] = Field(
        default_factory=list,
        description="Notable on-screen text elements that appeared (from diff).",
    )
    removed_elements: List[str] = Field(
        default_factory=list,
        description="Notable on-screen text elements that disappeared (from diff).",
    )


class FrameSummary(BaseModel):
    keyframe_idx: int
    frame_type: str
    t_sec: float
    timestamp: str
    image_path: str

    on_screen_text: List[str] = Field(default_factory=list)

    # NEW: all speakers present in this keyframe's utterances
    speakers: List[str] = Field(
        default_factory=list,
        description="Unique list of speakers who spoke during this keyframe (from assigned utterances).",
    )

    utterance_time_start: Optional[str] = None
    utterance_time_end: Optional[str] = None

    # UPDATED requirements: must explicitly mention speakers
    utterance_summary: str = Field(
        ...,
        description="Summary of utterances during this keyframe; must explicitly attribute statements to speakers.",
    )

    # More detailed
    content_summary: str = Field(
        ...,
        description="Detailed frame content summary grounded in frame_type, timestamp, on_screen_text, and screen_parse.",
    )

    # Combined synthesis
    combined_summary: str = Field(
        ...,
        description="Summary that combines utterance_summary and content_summary.",
    )

    # NEW: change summary for every keyframe transition (prev -> current). null for first keyframe.
    frame_change: Optional[FrameChange] = None

    similarity_to_prev: float = 0.0
    reused_prev_content: bool = False
    notes: List[str] = Field(default_factory=list)


class FinalOutput(BaseModel):
    meta: Dict[str, Any]
    keyframes: List[FrameSummary]


# -----------------------------
# History manager (diminishing returns)
# -----------------------------
@dataclass
class HistoryState:
    recent_frames: List[Dict[str, Any]]
    long_memory: str
    long_memory_max_chars: int = 4500

    def __init__(self):
        self.recent_frames = []
        self.long_memory = ""

    def add_frame(self, frame_summary_obj: Dict[str, Any], keep_recent: int = 4):
        self.recent_frames.append(frame_summary_obj)
        if len(self.recent_frames) > keep_recent:
            to_compress = self.recent_frames[:-keep_recent]
            self.recent_frames = self.recent_frames[-keep_recent:]
            return to_compress
        return []

    def build_history_context(self) -> str:
        parts = []
        if self.long_memory.strip():
            parts.append("LONG_MEMORY (old history, low weight):\n" + self.long_memory.strip())

        if self.recent_frames:
            parts.append("RECENT_HISTORY (high weight, most recent first):")
            for fr in reversed(self.recent_frames):
                parts.append(
                    f"- [{fr.get('timestamp','??')}] {fr.get('frame_type','?').upper()} "
                    f"combined_summary: {fr.get('combined_summary','')[:900]}"
                )
        return "\n".join(parts).strip()







def compress_into_long_memory(
    client: Groq,
    model: str,
    existing_long_memory: str,
    frames_to_compress: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    if not frames_to_compress:
        return existing_long_memory

    bullets = []
    for fr in frames_to_compress:
        bullets.append(
            f"[{fr.get('timestamp','??')}][{fr.get('frame_type','?')}] "
            f"{fr.get('combined_summary','')[:500]}"
        )
    chunk = "\n".join(bullets)

    system_instruction = (
        "You compress meeting history.\n"
        "Output must be short, factual, and useful.\n"
        "Do not invent details.\n"
        "Prefer concrete technical points and transitions.\n"
        "Keep the result under the requested character budget.\n"
        "Return ONLY plain text.\n"
    )

    user_prompt = (
        f"Existing LONG_MEMORY:\n{existing_long_memory}\n\n"
        f"New older frames to merge:\n{chunk}\n\n"
        f"Task:\n"
        f"1) Merge them into LONG_MEMORY.\n"
        f"2) Keep result <= {max_chars} characters.\n"
        f"3) Use bullet points.\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    text = response.choices[0].message.content.strip()

    if not text:
        merged = (existing_long_memory + "\n" + chunk).strip()
        return merged[:max_chars]

    return text[:max_chars]


# -----------------------------
# Core processing logic
# -----------------------------
def build_prompt_for_frame(
    frame: Dict[str, Any],
    history_context: str,
    prev_frame: Optional[Dict[str, Any]],
    prev_content_summary: Optional[str],
    similarity_to_prev: float,
    is_similar: bool,
    transition_diff: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    frame_type = (frame.get("frame_type") or "").lower()
    timestamp = frame.get("timestamp") or sec_to_hhmmss(frame.get("t_sec", 0.0))
    t_sec = float(frame.get("t_sec", 0.0))

    on_screen_text = frame.get("on_screen_text") or []
    screen_parse_summary = summarize_screen_parse(
        frame.get("screen_parse") or {},
        max_regions=8,
        max_region_lines=14,
        max_ocr_lines=140,
        max_chars=12000,
    )
    assigned_utterances = frame.get("assigned_utterances") or []
    speakers = extract_speakers_from_utterances(assigned_utterances)

    u_start_ts = None
    u_end_ts = None
    if assigned_utterances:
        u_start = min(float(u.get("_start_sec", u.get("start", t_sec))) for u in assigned_utterances)
        u_end = max(float(u.get("_end_sec", u.get("end", t_sec))) for u in assigned_utterances)
        u_start_ts = sec_to_hhmmss(u_start)
        u_end_ts = sec_to_hhmmss(u_end)

    utt_lines = []
    for u in assigned_utterances[:60]:
        s = float(u.get("_start_sec", u.get("start", 0.0)))
        e = float(u.get("_end_sec", u.get("end", 0.0)))
        spk = str(u.get("speaker", "unknown")).strip() or "unknown"
        txt = (u.get("text", "") or "").strip()
        utt_lines.append(f"[{sec_to_hhmmss(s)}-{sec_to_hhmmss(e)}][{spk}] {txt}")
    utterances_block = safe_join_text(utt_lines, max_chars=12000)

    reuse_instruction = ""
    if is_similar:
        reuse_instruction = (
            "IMPORTANT: This frame content is very similar to the previous keyframe.\n"
            "Do NOT repeat the entire explanation.\n"
            "Reuse prior context and focus on what is new.\n"
            "frame_change must still be filled if a previous keyframe exists.\n"
        )

    prev_block = ""
    prev_content_summary_block = "PREVIOUS_KEYFRAME_CONTENT_SUMMARY:\nnone\n\n"
    if prev_frame is not None:
        prev_idx = prev_frame.get("keyframe_idx", -1)
        prev_ts = prev_frame.get("timestamp") or sec_to_hhmmss(prev_frame.get("t_sec", 0.0))
        prev_type = (prev_frame.get("frame_type") or "unknown").lower()
        prev_block = (
            "PREVIOUS_KEYFRAME:\n"
            f"- keyframe_idx: {prev_idx}\n"
            f"- frame_type: {prev_type}\n"
            f"- timestamp: {prev_ts}\n\n"
        )
        prev_content_summary_block = (
            "PREVIOUS_KEYFRAME_CONTENT_SUMMARY:\n"
            f"{(prev_content_summary or 'unknown').strip()}\n\n"
        )

    transition_diff_block = ""
    if transition_diff is not None:
        transition_diff_block = (
            "KEYFRAME_TRANSITION_DIFF (computed from on_screen_text):\n"
            f"added_elements: {transition_diff.get('added_elements', [])}\n"
            f"removed_elements: {transition_diff.get('removed_elements', [])}\n\n"
        )

    system_instruction = (
        "You are generating time-aware meeting notes per keyframe.\n"
        "You must follow the provided schema exactly and return JSON only.\n"
        "Do not invent facts not present in the inputs.\n"
        "If something is unknown, say unknown.\n"
        "History has diminishing importance: RECENT_HISTORY is high weight, LONG_MEMORY is low weight.\n"
        "Speaker attribution is required for utterance summary.\n"
    )

    on_screen_capped = on_screen_text[:350]

    if frame_type == "slides":
        content_task = (
            "For slides:\n"
            "- content_summary must use frame_type + timestamp + on_screen_text + screen_parse.\n"
            "  Cover headings, bullets, numbers, claims, and relationships visible on screen.\n"
            "- combined_summary must combine utterance_summary + content_summary.\n"
        )
    elif frame_type == "code":
        content_task = (
            "For code:\n"
            "- content_summary must use frame_type + timestamp + on_screen_text + screen_parse.\n"
            "  Cover files/modules, functions/classes, logic, inputs/outputs, and config if visible.\n"
            "- combined_summary must combine utterance_summary + content_summary.\n"
        )
    else:
        content_task = (
            "For demo:\n"
            "- content_summary must use frame_type + timestamp + on_screen_text + screen_parse.\n"
            "  Cover screens, controls, state transitions, and resulting behavior.\n"
            "- combined_summary must combine utterance_summary + content_summary.\n"
        )

    output_rules = (
        "OUTPUT_RULES (must follow exactly):\n"
        "- Always populate: on_screen_text, speakers, utterance_summary, content_summary, combined_summary.\n"
        "- utterance_summary must use utterance timestamps + speaker + text provided.\n"
        "- content_summary must be grounded in frame_type + timestamp + on_screen_text + screen_parse.\n"
        "- combined_summary must summarize utterance_summary and content_summary.\n"
        "- If previous keyframe exists, frame_change must be present.\n"
        "  - changed_summary must be only the difference between previous and current content_summary.\n"
        "  - possible_reason remains grounded in utterances/on-screen evidence; else unknown.\n"
        "  - added_elements and removed_elements must use provided diff lists.\n"
        "- If no previous keyframe exists, frame_change must be null.\n"
    )

    user_prompt = (
        f"{prev_block}"
        f"CURRENT_KEYFRAME:\n"
        f"- keyframe_idx: {frame.get('keyframe_idx')}\n"
        f"- frame_type: {frame_type}\n"
        f"- t_sec: {t_sec}\n"
        f"- timestamp: {timestamp}\n"
        f"- image_path: {frame.get('image_path')}\n"
        f"- similarity_to_prev: {similarity_to_prev:.3f}\n"
        f"- detected_speakers: {speakers}\n"
        f"- utterance_time_range: {u_start_ts}-{u_end_ts}\n\n"
        f"ON_SCREEN_TEXT (list):\n{on_screen_capped}\n\n"
        f"SCREEN_PARSE (structured parse of current frame):\n{screen_parse_summary}\n\n"
        f"ASSIGNED_UTTERANCES (time-stamped, includes speaker):\n{utterances_block}\n\n"
        f"{transition_diff_block}"
        f"{prev_content_summary_block}"
        f"HISTORY_CONTEXT:\n{history_context}\n\n"
        f"{output_rules}\n\n"
        f"{reuse_instruction}\n"
        f"{content_task}\n"
        f"Now produce the JSON output for this keyframe following the schema."
    )

    return system_instruction, user_prompt


def keyframe_items(keyframes_data: Any) -> List[Dict[str, Any]]:
    if isinstance(keyframes_data, dict):
        return keyframes_data.get("keyframes", []) or []
    if isinstance(keyframes_data, list):
        return keyframes_data
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True, help="Path to keyframes_with_utterances.json")
    ap.add_argument("--out", required=True, help="Output path for final JSON")
    ap.add_argument("--model", default="llama3-70b-8192", help="Gemini model id")
    ap.add_argument("--similarity_threshold", type=float, default=0.82, help="Similarity threshold for 'reuse prev content'")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    log("Starting build_final_output.py ...")
    log(f"Keyframes file: {args.keyframes}")
    log(f"Output file: {args.out}")
    log(f"Model: {args.model}")

    keyframes_data = load_json(args.keyframes)
    keyframes_list = keyframe_items(keyframes_data)
    if not keyframes_list:
        raise ValueError("No keyframes found in input keyframes file.")

    # Process keyframes in chronological order.
    keyframes_list = sorted(
        keyframes_list,
        key=lambda x: (
            float(x.get("t_sec", 0.0)),
            int(x.get("keyframe_idx", 0)),
        ),
    )

    log(f"Loaded keyframes: {len(keyframes_list)}")

    log("Initializing Groq client...")    client = groq_client()
    log("Gemini client ready.")

    output = {
        "meta": {
            "keyframes_file": args.keyframes,
            "model": args.model,
            "generated_at_epoch": time.time(),
            "rules": {
                "process_order": "keyframes in chronological order",
                "history": "recent detailed + long_memory compressed (diminishing returns)",
                "similarity_threshold": args.similarity_threshold,
                "transition_change_each_keyframe": True,
                "speakers_per_keyframe": True,
                "utterance_summary_requires_speaker_attribution": True,
                "content_summary_uses_screen_parse": True,
                "combined_summary_synthesizes_utterance_and_content": True,
                "change_summary_is_content_diff": True,
            },
        },
        "keyframes": [],
    }

    history_state = HistoryState()

    prev_frame_obj: Optional[Dict[str, Any]] = None
    prev_frame_summary: Optional[Dict[str, Any]] = None

    global_kf_done = 0
    global_kf_total = len(keyframes_list)
    log(f"Total keyframes to process: {global_kf_total}")

    for frame in keyframes_list:
        global_kf_done += 1
        kf_idx = frame.get("keyframe_idx")
        kf_ts = frame.get("timestamp") or sec_to_hhmmss(frame.get("t_sec", 0.0))
        kf_type = (frame.get("frame_type") or "unknown").lower()
        utt_count = len(frame.get("assigned_utterances") or [])
        log(f"[{global_kf_done}/{global_kf_total}] Keyframe {kf_idx} @ {kf_ts} | type={kf_type} | utterances={utt_count}")

        sig_cur = frame_signature(frame)
        sig_prev = frame_signature(prev_frame_obj)
        sim = jaccard_similarity(sig_prev, sig_cur) if prev_frame_obj else 0.0
        is_similar = (prev_frame_obj is not None) and (sim >= args.similarity_threshold)
        log(f"  similarity_to_prev={sim:.3f} | reused_prev_content={is_similar}")

        transition_diff = None
        if prev_frame_obj is not None:
            prev_text = (prev_frame_obj.get("on_screen_text") or [])
            cur_text = (frame.get("on_screen_text") or [])
            added, removed = diff_lists(prev_text, cur_text, max_items=40)
            transition_diff = {"added_elements": added, "removed_elements": removed}

        history_context = history_state.build_history_context()

        system_instruction, user_prompt = build_prompt_for_frame(
            frame=frame,
            history_context=history_context,
            prev_frame=prev_frame_obj,
            prev_content_summary=(prev_frame_summary or {}).get("content_summary"),
            similarity_to_prev=sim,
            is_similar=is_similar,
            transition_diff=transition_diff,
        )

        log("  -> Calling Gemini ...")
        t_call = time.time()


        parsed = call_llm_structured(
            client=client,
            model=args.model,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            temperature=args.temperature,
            max_retries=3,
        )


        log(f"  <- Gemini done in {time.time() - t_call:.1f}s")

        parsed_dict = dict(parsed)

        parsed_dict["similarity_to_prev"] = float(sim)
        parsed_dict["reused_prev_content"] = bool(is_similar)
        if "notes" not in parsed_dict:
            parsed_dict["notes"] = []
        if is_similar:
            parsed_dict["notes"].append("High similarity to previous keyframe; instructed incremental update.")
        if prev_frame_summary is not None:
            parsed_dict["notes"].append("Keyframe-to-keyframe transition diff computed and provided (frame_change required).")

        # Enforce change summary as strict diff of previous vs current content_summary.
        if prev_frame_summary is None:
            parsed_dict["frame_change"] = None
        else:
            prev_content_summary = (prev_frame_summary or {}).get("content_summary")
            current_content_summary = parsed_dict.get("content_summary")
            existing_change = parsed_dict.get("frame_change") or {}
            if not isinstance(existing_change, dict):
                existing_change = {}
            existing_change["changed_summary"] = build_content_change_summary(
                prev_content_summary=prev_content_summary,
                cur_content_summary=current_content_summary,
            )
            existing_change["possible_reason"] = str(existing_change.get("possible_reason", "")).strip() or "unknown"
            existing_change["added_elements"] = (transition_diff or {}).get("added_elements", [])
            existing_change["removed_elements"] = (transition_diff or {}).get("removed_elements", [])
            parsed_dict["frame_change"] = existing_change

        output["keyframes"].append(parsed_dict)

        to_compress = history_state.add_frame(
            frame_summary_obj={
                "timestamp": parsed_dict.get("timestamp"),
                "frame_type": parsed_dict.get("frame_type"),
                "combined_summary": parsed_dict.get("combined_summary", ""),
            },
            keep_recent=4,
        )
        if to_compress:
            log(f"  -> Compressing {len(to_compress)} older frame(s) into LONG_MEMORY ...")
            history_state.long_memory = compress_into_long_memory(
                client=client,
                model=args.model,
                existing_long_memory=history_state.long_memory,
                frames_to_compress=to_compress,
                max_chars=history_state.long_memory_max_chars,
            )
            log("  <- LONG_MEMORY updated.")

        prev_frame_obj = frame
        prev_frame_summary = parsed_dict

    log("\nAll keyframes processed. Writing output JSON ...")
    save_json(args.out, output)
    log(f"Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()



