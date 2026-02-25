# condense_final_output.py
# Usage:
#   python condense_final_output.py --in "C:\meet-agent\out_folder\final_output.json" --out "C:\meet-agent\out_folder\final_output_condensed.json"
#
# What it does:
# - Reads the "final_output.json" produced by your build script
# - Produces a condensed version with only:
#     - keyframe (idx, timestamp, type, t_sec, image_path)
#     - combined_summary
#     - changed_summary (from transition_change/frame_change/demo_change if present)
# - Supports both input schemas:
#     1) new: {"meta": ..., "keyframes": [...]}
#     2) old: {"meta": ..., "topics": [{"keyframes": [...]}]}

import argparse
import json
import os
from typing import Any, Dict, Optional


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pick_changed_summary(kf: Dict[str, Any]) -> Optional[str]:
    """
    Tries multiple locations, because your schema may store change summaries under different keys
    depending on how you implemented transitions.

    Priority order:
      1) transition_change.changed_summary
      2) frame_change.changed_summary
      3) demo_change.changed_summary
      4) changed_summary at root (fallback)
    """
    for container_key in ("transition_change", "frame_change", "demo_change"):
        container = kf.get(container_key)
        if isinstance(container, dict):
            cs = container.get("changed_summary")
            if isinstance(cs, str) and cs.strip():
                return cs.strip()

    cs_root = kf.get("changed_summary")
    if isinstance(cs_root, str) and cs_root.strip():
        return cs_root.strip()

    return None


def condense_keyframe(kf: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "keyframe": {
            "keyframe_idx": kf.get("keyframe_idx"),
            "timestamp": kf.get("timestamp"),
            "frame_type": kf.get("frame_type"),
            "t_sec": kf.get("t_sec"),
            "image_path": kf.get("image_path"),
        },
        "combined_summary": kf.get("combined_summary"),
        "changed_summary": pick_changed_summary(kf),
    }


def condense(final_obj: Dict[str, Any]) -> Dict[str, Any]:
    out_meta: Dict[str, Any] = {
        "source": final_obj.get("meta", {}),
        "notes": "Condensed output: keyframe + combined_summary + changed_summary",
    }

    # New schema: root keyframes list
    root_keyframes = final_obj.get("keyframes", [])
    if isinstance(root_keyframes, list):
        out: Dict[str, Any] = {
            "meta": {**out_meta, "input_schema": "root_keyframes"},
            "keyframes": [],
        }
        for kf in root_keyframes:
            if not isinstance(kf, dict):
                continue
            out["keyframes"].append(condense_keyframe(kf))
        return out

    # Old schema: topics[] with keyframes[]
    out = {
        "meta": {**out_meta, "input_schema": "topics"},
        "topics": [],
    }

    topics = final_obj.get("topics", [])
    if not isinstance(topics, list):
        topics = []

    for t in topics:
        if not isinstance(t, dict):
            continue

        topic_out = {
            "topic": t.get("topic"),
            "start": t.get("start"),
            "end": t.get("end"),
            "start_ts": t.get("start_ts"),
            "end_ts": t.get("end_ts"),
            "keyframes": [],
        }

        keyframes = t.get("keyframes", [])
        if not isinstance(keyframes, list):
            keyframes = []

        for kf in keyframes:
            if not isinstance(kf, dict):
                continue
            topic_out["keyframes"].append(condense_keyframe(kf))

        out["topics"].append(topic_out)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to final_output.json")
    ap.add_argument("--out", dest="out", required=True, help="Path to write condensed JSON")
    args = ap.parse_args()

    final_obj = load_json(args.inp)
    if not isinstance(final_obj, dict):
        raise ValueError("Input JSON root must be an object/dict (expected FinalOutput-like structure).")

    condensed = condense(final_obj)
    save_json(args.out, condensed)
    print(f"Wrote condensed JSON: {args.out}")


if __name__ == "__main__":
    main()
