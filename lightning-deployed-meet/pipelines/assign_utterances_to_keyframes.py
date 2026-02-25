import json
import argparse
from typing import Any, Dict, List, Optional, Tuple


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def extract_list(data: Any) -> List[Dict[str, Any]]:
    # Accept either a list of items, or a dict that contains a list under common keys.
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ["utterances", "items", "segments", "results", "data"]:
            if k in data and isinstance(data[k], list):
                return [x for x in data[k] if isinstance(x, dict)]
    return []


def extract_keyframes(data: Any) -> List[Dict[str, Any]]:
    # Accept either a list of keyframes, or a dict that contains a list under common keys.
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ["keyframes", "items", "results", "data"]:
            if k in data and isinstance(data[k], list):
                return [x for x in data[k] if isinstance(x, dict)]
    return []


def get_time_field(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            try:
                v = d[k]
                if v is None:
                    continue
                return float(v)
            except Exception:
                continue
    return None


def get_utterance_times(u: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    # Try common fields for start/end times
    start = get_time_field(u, ["start_sec", "start_s", "start", "start_time", "t_start", "begin", "from"])
    end = get_time_field(u, ["end_sec", "end_s", "end", "end_time", "t_end", "finish", "to"])

    # If only one is present, treat utterance as a point-in-time
    if start is not None and end is None:
        end = start
    if end is not None and start is None:
        start = end

    return start, end


def get_utterance_text(u: Dict[str, Any]) -> str:
    for k in ["text", "utterance", "content", "transcript", "sentence"]:
        if k in u and safe_str(u[k]).strip():
            return safe_str(u[k]).strip()

    # Some formats store words list
    if "words" in u and isinstance(u["words"], list):
        parts = []
        for w in u["words"]:
            if isinstance(w, dict):
                t = w.get("word") or w.get("text")
                if t:
                    parts.append(str(t))
            elif isinstance(w, str):
                parts.append(w)
        if parts:
            return " ".join(parts).strip()

    return ""


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    # Closed-open overlap check: [a0, a1) overlaps [b0, b1) iff max(starts) < min(ends)
    return max(a0, b0) < min(a1, b1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keyframes_json", help="Path to keyframes JSON (e.g. keyframes_parsed.json)")
    ap.add_argument("utterances_json", help="Path to utterances.json")
    ap.add_argument("-o", "--out", default="keyframes_with_utterances.json", help="Output JSON path")
    ap.add_argument(
        "--pre-roll-sec",
        type=float,
        default=3.0,
        help="Seconds before each keyframe start that should also belong to that keyframe.",
    )
    args = ap.parse_args()

    # Load keyframes
    with open(args.keyframes_json, "r", encoding="utf-8") as f:
        kf_raw = json.load(f)

    keyframes_list = extract_keyframes(kf_raw)
    if not keyframes_list:
        raise ValueError(
            "No keyframes found. Expected a list, or an object containing keyframes under one of: "
            "keyframes/items/results/data."
        )

    # Sort keyframes by time
    keyframes = sorted(
        keyframes_list,
        key=lambda k: (
            float(k.get("t_sec", 0.0) or 0.0),
            int(k.get("keyframe_idx", 0) or 0),
        ),
    )
    if not keyframes:
        raise ValueError("No keyframes found in keyframes JSON")

    pre_roll_sec = max(0.0, float(args.pre_roll_sec))

    # Precompute keyframe times and windows.
    # window i:
    # - first keyframe: [t_0, t_1)
    # - others: [max(t_i - pre_roll_sec, t_{i-1}), t_{i+1})
    # This makes [t_i - pre_roll_sec, t_i) belong to BOTH keyframe i and keyframe i-1.
    t = [float(kf.get("t_sec", 0.0) or 0.0) for kf in keyframes]
    n = len(t)
    windows: List[Tuple[float, float]] = []
    for i in range(n):
        if i == 0:
            start = t[i]
        else:
            start = max(t[i] - pre_roll_sec, t[i - 1])
        end = t[i + 1] if i < n - 1 else float("inf")
        windows.append((start, end))

    # Prepare output keyframes (copy + add assigned_utterances)
    out_keyframes: List[Dict[str, Any]] = []
    for kf in keyframes:
        kf_out = dict(kf)
        kf_out["assigned_utterances"] = []
        out_keyframes.append(kf_out)

    # Load utterances
    with open(args.utterances_json, "r", encoding="utf-8") as f:
        u_raw = json.load(f)

    utterances = extract_list(u_raw)
    if not utterances:
        raise ValueError(
            "No utterances found. Expected utterances.json to be a list, or a dict containing a list under "
            "one of: utterances/items/segments/results/data."
        )

    unassigned = []
    multi_assigned = 0
    assigned_total = 0

    for u in utterances:
        text = get_utterance_text(u).strip()
        u_start, u_end = get_utterance_times(u)

        if u_start is None or u_end is None or not text:
            unassigned.append({"reason": "missing_text_or_time", "utterance": u})
            continue

        u_start = float(u_start)
        u_end = float(u_end)
        if u_end < u_start:
            u_start, u_end = u_end, u_start

        # Make point-in-time utterances half-open with tiny duration
        if u_end == u_start:
            u_end = u_start + 1e-6

        matched_indexes = []
        for i, (w0, w1) in enumerate(windows):
            if overlaps(u_start, u_end, w0, w1):
                matched_indexes.append(i)

        if not matched_indexes:
            # Fallback for degenerate boundary conditions.
            for i, (w0, w1) in enumerate(windows):
                eps = 1e-9
                if overlaps(u_start - eps, u_end + eps, w0, w1):
                    matched_indexes.append(i)

        if not matched_indexes:
            unassigned.append({"reason": "no_overlapping_keyframe_window", "utterance": u})
            continue

        # Keep indexes sorted and unique.
        matched_indexes = sorted(set(matched_indexes))

        if len(matched_indexes) > 1:
            multi_assigned += 1

        payload = dict(u)
        payload["_text"] = text
        payload["_start_sec"] = u_start
        payload["_end_sec"] = u_end
        payload["_overlaps_sorted_indexes"] = matched_indexes

        for idx in matched_indexes:
            payload2 = dict(payload)
            payload2["_assigned_sorted_index"] = idx
            payload2["_assigned_keyframe_idx"] = out_keyframes[idx].get("keyframe_idx")
            payload2["_assigned_t_sec"] = out_keyframes[idx].get("t_sec")
            out_keyframes[idx]["assigned_utterances"].append(payload2)
            assigned_total += 1

    # Sort utterances inside each keyframe by start time
    for kf in out_keyframes:
        kf["assigned_utterances"].sort(key=lambda x: float(x.get("_start_sec", 0.0) or 0.0))

    out = {
        "meta": {
            "keyframes_file": args.keyframes_json,
            "utterances_file": args.utterances_json,
            "keyframes_count": len(out_keyframes),
            "utterances_count": len(utterances),
            "assigned_total": assigned_total,  # counts duplicates if an utterance overlaps multiple keyframes
            "multi_assigned_utterances": multi_assigned,
            "unassigned_count": len(unassigned),
            "pre_roll_sec": pre_roll_sec,
            "window_strategy": (
                "pre-roll overlap windows: "
                "first [t_0, t_1), others [max(t_i-pre_roll_sec, t_{i-1}), t_{i+1}), "
                "last ends at +inf"
            ),
        },
        "keyframes": out_keyframes,
        "unassigned_utterances": unassigned,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote: {args.out}")
    print(f"Keyframes: {len(out_keyframes)}")
    print(f"Utterances: {len(utterances)}")
    print(f"Assigned total (including duplicates): {assigned_total}")
    print(f"Utterances that overlapped multiple keyframes: {multi_assigned}")
    print(f"Unassigned utterances: {len(unassigned)}")


if __name__ == "__main__":
    main()
