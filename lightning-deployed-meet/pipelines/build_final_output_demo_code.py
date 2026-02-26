#!/usr/bin/env python3
"""
Demo-only Gemini build stage (kept in demo-code route for compatibility).

Behavior:
- `demo` keyframes: summarized with Gemini.
- `slides`, `code`, and `none` keyframes: NO Gemini call; output is built from OCR + utterances.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq


def log(msg: str) -> None:
    print(msg, flush=True)


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
    return [t for t in s.split() if t]


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def safe_join_text(lines: List[str], max_chars: int = 8000) -> str:
    out = []
    total = 0
    for ln in lines:
        if total + len(ln) + 1 > max_chars:
            break
        out.append(ln)
        total += len(ln) + 1
    return "\n".join(out)


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def build_content_change_summary(
    prev_content_summary: Optional[Any],
    cur_content_summary: Optional[Any],
    max_items: int = 6,
) -> str:

    if isinstance(prev_content_summary, dict):
        prev_content_summary = json.dumps(prev_content_summary)

    if isinstance(cur_content_summary, dict):
        cur_content_summary = json.dumps(cur_content_summary)

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


def frame_signature(frame: Optional[Dict[str, Any]]) -> str:
    if not frame:
        return ""
    on_screen = frame.get("on_screen_text") or []
    return safe_join_text([str(x) for x in on_screen[:120]], max_chars=3000)


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


def extract_speakers_from_utterances(utterances: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    out = []
    for u in utterances or []:
        spk = str(u.get("speaker", "")).strip() or "unknown"
        if spk not in seen:
            seen.add(spk)
            out.append(spk)
    return out


def utterance_time_bounds(utterances: List[Dict[str, Any]], default_t: float) -> Tuple[Optional[str], Optional[str]]:
    if not utterances:
        return None, None
    starts = []
    ends = []
    for u in utterances:
        try:
            starts.append(float(u.get("_start_sec", u.get("start", default_t))))
            ends.append(float(u.get("_end_sec", u.get("end", default_t))))
        except Exception:
            continue
    if not starts or not ends:
        return None, None
    return sec_to_hhmmss(min(starts)), sec_to_hhmmss(max(ends))


def build_utterance_lines(utterances: List[Dict[str, Any]], max_lines: int = 80) -> List[str]:
    lines: List[str] = []
    for u in utterances[:max_lines]:
        try:
            s = float(u.get("_start_sec", u.get("start", 0.0)))
            e = float(u.get("_end_sec", u.get("end", 0.0)))
        except Exception:
            s, e = 0.0, 0.0
        spk = str(u.get("speaker", "unknown")).strip() or "unknown"
        txt = (u.get("text", "") or "").strip()
        if not txt:
            continue
        lines.append(f"[{sec_to_hhmmss(s)}-{sec_to_hhmmss(e)}][{spk}] {txt}")
    return lines


def local_summary_for_non_demo(frame: Dict[str, Any]) -> Dict[str, str]:
    frame_type = str(frame.get("frame_type", "unknown")).lower()
    ocr_lines = [str(x).strip() for x in (frame.get("on_screen_text") or []) if str(x).strip()]
    utter_lines = build_utterance_lines(frame.get("assigned_utterances") or [], max_lines=20)

    if utter_lines:
        utterance_summary = " | ".join(utter_lines[:8])
    else:
        utterance_summary = "No assigned utterances for this keyframe."

    if ocr_lines:
        content_summary = (
            f"{frame_type.upper()} keyframe. OCR extracted on-screen text (top lines): "
            + " | ".join(ocr_lines[:25])
        )
    else:
        content_summary = f"{frame_type.upper()} keyframe. OCR text not available."

    combined_summary = (
        f"Local (no Gemini) summary for {frame_type} frame. "
        f"Utterances: {utterance_summary} "
        f"Content: {content_summary}"
    )

    return {
        "utterance_summary": utterance_summary,
        "content_summary": content_summary,
        "combined_summary": combined_summary,
    }



def groq_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment.")
    return Groq(api_key=api_key)


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

            try:
                return json.loads(text_output)
            except Exception:
                return {
                    "utterance_summary": "",
                    "content_summary": text_output[:1500],
                    "combined_summary": text_output[:1500],
                }
        except Exception as e:
            last_err = e
            time.sleep(0.7 * attempt)

    raise RuntimeError(f"Groq structured call failed after retries: {last_err}")


def build_demo_prompt(
    frame: Dict[str, Any],
    prev_content_summary: Optional[str],
    similarity_to_prev: float,
    is_similar: bool,
) -> Tuple[str, str]:
    frame_type = str(frame.get("frame_type", "unknown")).lower()
    timestamp = frame.get("timestamp") or sec_to_hhmmss(frame.get("t_sec", 0.0))
    t_sec = float(frame.get("t_sec", 0.0))
    on_screen_text = frame.get("on_screen_text") or []
    screen_parse_summary = summarize_screen_parse(frame.get("screen_parse") or {})
    utterances_block = safe_join_text(
        build_utterance_lines(frame.get("assigned_utterances") or [], max_lines=80),
        max_chars=12000,
    )
    reuse_instruction = ""
    if is_similar:
        reuse_instruction = (
            "Frame is highly similar to previous keyframe. Reuse context and focus on what changed.\n"
        )

    prev_block = "PREVIOUS_KEYFRAME_CONTENT_SUMMARY:\nnone\n"
    if prev_content_summary:
        prev_block = f"PREVIOUS_KEYFRAME_CONTENT_SUMMARY:\n{prev_content_summary}\n"

    system_instruction = (
        "You generate keyframe-level meeting notes for demo screens only.\n"
        "Ground all claims in provided utterances and OCR/screen parse.\n"
        "Do not invent facts.\n"
        "Return ONLY valid JSON.\n"
        "Do NOT include markdown.\n"
        "Do NOT include backticks.\n"
    )

    user_prompt = (
        f"CURRENT_KEYFRAME:\n"
        f"- frame_type: {frame_type}\n"
        f"- keyframe_idx: {frame.get('keyframe_idx')}\n"
        f"- t_sec: {t_sec}\n"
        f"- timestamp: {timestamp}\n"
        f"- image_path: {frame.get('image_path')}\n"
        f"- similarity_to_prev: {similarity_to_prev:.3f}\n\n"
        f"ON_SCREEN_TEXT:\n{on_screen_text[:350]}\n\n"
        f"SCREEN_PARSE:\n{screen_parse_summary}\n\n"
        f"ASSIGNED_UTTERANCES:\n{utterances_block}\n\n"
        f"{prev_block}\n"
        f"{reuse_instruction}\n"
        f"Requirements:\n"
        f"- utterance_summary: attribute statements to speakers when present.\n"
        f"- content_summary: describe what is visible/changed in this frame.\n"
        f"- combined_summary: merge utterance + visual context.\n"
    )
    return system_instruction, user_prompt


def keyframe_items(keyframes_data: Any) -> List[Dict[str, Any]]:
    if isinstance(keyframes_data, dict):
        return keyframes_data.get("keyframes", []) or []
    if isinstance(keyframes_data, list):
        return keyframes_data
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True, help="Path to keyframes_with_utterances.json")
    ap.add_argument("--out", required=True, help="Output path for final JSON")
    ap.add_argument("--model", default="llama3-70b-8192", help="Groq model id")
    ap.add_argument("--similarity-threshold", type=float, default=0.82)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    keyframes_data = load_json(args.keyframes)
    keyframes_list = keyframe_items(keyframes_data)
    if not keyframes_list:
        raise ValueError("No keyframes found in input keyframes file.")

    keyframes_list = sorted(
        keyframes_list,
        key=lambda x: (float(x.get("t_sec", 0.0)), int(x.get("keyframe_idx", 0))),
    )

    demo_count = sum(1 for kf in keyframes_list if str(kf.get("frame_type", "")).lower() == "demo")
    code_count = sum(1 for kf in keyframes_list if str(kf.get("frame_type", "")).lower() == "code")
    gemini_target_count = demo_count
    local_only_count = len(keyframes_list) - gemini_target_count
    log(
        f"Loaded keyframes: total={len(keyframes_list)} demo={demo_count} "
        f"code={code_count} local_only={local_only_count}"
    )

    client: Optional[Groq] = None
    if gemini_target_count > 0:
        log("Initializing Groq client (demo frames only)...")
        client = groq_client()
        log("Groq client ready.")

    output: Dict[str, Any] = {
        "meta": {
            "keyframes_file": args.keyframes,
            "model": args.model,
            "generated_at_epoch": time.time(),
            "rules": {
                "demo_frames_use_llm": True,
                "slides_code_none_use_local_ocr_only": True,
                "similarity_threshold": args.similarity_threshold,
                "frame_change_is_deterministic_content_diff": True,
            },
            "counts": {
                "total_keyframes": len(keyframes_list),
                "demo_keyframes": demo_count,
                "code_keyframes": code_count,
                "llm_keyframes": gemini_target_count,

                "local_only_keyframes": local_only_count,
                "llm_calls": 0,
            },
        },
        "keyframes": [],
    }

    prev_frame_obj: Optional[Dict[str, Any]] = None
    prev_content_summary: Optional[str] = None

    for idx, frame in enumerate(keyframes_list, start=1):
        frame_type = str(frame.get("frame_type", "unknown")).lower()
        t_sec = float(frame.get("t_sec", 0.0))
        timestamp = frame.get("timestamp") or sec_to_hhmmss(t_sec)
        on_screen_text = [str(x).strip() for x in (frame.get("on_screen_text") or []) if str(x).strip()]
        assigned_utterances = frame.get("assigned_utterances") or []
        speakers = extract_speakers_from_utterances(assigned_utterances)
        utt_start_ts, utt_end_ts = utterance_time_bounds(assigned_utterances, default_t=t_sec)

        sim = 0.0
        is_similar = False
        if prev_frame_obj is not None:
            sim = jaccard_similarity(frame_signature(prev_frame_obj), frame_signature(frame))
            is_similar = sim >= float(args.similarity_threshold)

        log(
            f"[{idx}/{len(keyframes_list)}] keyframe={frame.get('keyframe_idx')} "
            f"type={frame_type} time={timestamp} similarity={sim:.3f}"
        )

        if frame_type == "demo":
            if client is None:
                raise RuntimeError("Internal error: demo frame encountered but Gemini client is not initialized.")
            system_instruction, user_prompt = build_demo_prompt(
                frame=frame,
                prev_content_summary=prev_content_summary,
                similarity_to_prev=sim,
                is_similar=is_similar,
            )
            t0 = time.time()
            parsed = call_llm_structured(
                client=client,
                model=args.model,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_retries=3,
            )
            log(f"  Groq done in {time.time() - t0:.1f}s")
            output["meta"]["counts"]["llm_calls"] += 1
            
            summary_payload = parsed


            summary_source = "groq_demo_only"
        else:
            summary_payload = local_summary_for_non_demo(frame)
            summary_source = "local_ocr_only"

        transition_diff = {"added_elements": [], "removed_elements": []}
        if prev_frame_obj is not None:
            prev_text = [str(x).strip() for x in (prev_frame_obj.get("on_screen_text") or []) if str(x).strip()]
            cur_text = on_screen_text
            added, removed = diff_lists(prev_text, cur_text, max_items=40)
            transition_diff = {"added_elements": added, "removed_elements": removed}

        frame_change = None
        if prev_content_summary is not None:
            frame_change = {
                "changed_summary": build_content_change_summary(
                    prev_content_summary=prev_content_summary,
                    cur_content_summary=summary_payload.get("content_summary"),
                ),
                "possible_reason": (
                    "Computed from keyframe OCR and utterance differences; no transition LLM call used."
                ),
                "added_elements": transition_diff["added_elements"],
                "removed_elements": transition_diff["removed_elements"],
            }

        out_frame = {
            "keyframe_idx": int(frame.get("keyframe_idx", idx - 1)),
            "frame_type": frame_type,
            "t_sec": t_sec,
            "timestamp": timestamp,
            "image_path": str(frame.get("image_path", "")),
            "on_screen_text": on_screen_text[:400],
            "speakers": speakers,
            "utterance_time_start": utt_start_ts,
            "utterance_time_end": utt_end_ts,
            "utterance_summary": str(summary_payload.get("utterance_summary", "")).strip(),
            "content_summary": str(summary_payload.get("content_summary", "")).strip(),
            "combined_summary": str(summary_payload.get("combined_summary", "")).strip(),
            "frame_change": frame_change,
            "similarity_to_prev": float(sim),
            "reused_prev_content": bool(is_similar and frame_type == "demo"),
            "notes": [
                f"summary_source={summary_source}",
                "Only demo keyframes are sent to Gemini in this pipeline.",
            ],
        }

        output["keyframes"].append(out_frame)
        prev_frame_obj = frame
        prev_content_summary = out_frame.get("content_summary")

    save_json(args.out, output)
    log(f"Done. Wrote: {args.out}")
    log(f"LLM calls made: {output['meta']['counts']['llm_calls']}")


if __name__ == "__main__":
    main()
