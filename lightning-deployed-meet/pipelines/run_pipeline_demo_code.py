#!/usr/bin/env python3
"""
Demo-only Gemini pipeline orchestrator (kept in demo-code route for compatibility).

Pipeline steps:
1) deepgram_extract_utterances.py         (parallel)
2) smart_keyframes_and_classify.py        (parallel)
3) assign_utterances_to_keyframes.py
4) build_final_output_demo_code.py        (Gemini for demo only; slides+code local OCR/transcript)
5) condense_final_output.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple


def run_command(name: str, cmd: Sequence[str], cwd: Path) -> None:
    start = time.perf_counter()
    print(f"\n[{name}] START")
    print(f"[{name}] CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    dur = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"[{name}] failed with exit code {result.returncode}")
    print(f"[{name}] DONE in {dur:.2f}s")


def run_parallel(commands: List[Tuple[str, List[str]]], cwd: Path) -> None:
    if not commands:
        return
    with ThreadPoolExecutor(max_workers=len(commands)) as ex:
        futures = {ex.submit(run_command, name, cmd, cwd): name for name, cmd in commands}
        for fut in as_completed(futures):
            fut.result()


def require_file(path: Path, step_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[{step_name}] expected output not found: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run demo-only LLM meeting pipeline (demo-code route alias).")
    ap.add_argument("--video", required=True, help="Path to meeting video/audio input.")
    ap.add_argument("--out", required=True, help="Output directory for pipeline artifacts.")

    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")

    ap.add_argument("--deepgram-model", default="nova-3", help="Deepgram model.")
    ap.add_argument("--deepgram-language", default=None, help="Deepgram language (optional).")
    ap.add_argument(
        "--deepgram-raw-out",
        default=None,
        help="Optional path for raw Deepgram response JSON.",
    )
    ap.add_argument(
        "--deepgram-request-timeout-sec",
        type=float,
        default=1200.0,
        help="HTTP request timeout for Deepgram call.",
    )
    ap.add_argument(
        "--deepgram-connect-timeout-sec",
        type=float,
        default=30.0,
        help="HTTP connect timeout for Deepgram call.",
    )
    ap.add_argument(
        "--deepgram-retries",
        type=int,
        default=3,
        help="Retry attempts for Deepgram call.",
    )
    ap.add_argument(
        "--deepgram-retry-backoff-sec",
        type=float,
        default=2.0,
        help="Base retry backoff seconds for Deepgram call.",
    )
    ap.add_argument(
        "--force-deepgram",
        action="store_true",
        help="Re-run Deepgram even if utterances.json already exists.",
    )

    ap.add_argument("--force-keyframes", action="store_true", help="Pass --force to smart keyframe script.")
    ap.add_argument("--pre-roll-sec", type=float, default=3.0, help="Pre-roll seconds for utterance assignment.")

    ap.add_argument("--llm-model", default="llama3-70b-8192", help="LLM model id.")
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for demo prompt reuse logic.",
    )
    ap.add_argument("--temperature", type=float, default=0.2, help="Gemini temperature for demo keyframes.")
    args = ap.parse_args()

    pipeline_dir = Path(__file__).resolve().parent
    repo_dir = pipeline_dir

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    deepgram_script = repo_dir / "deepgram_extract_utterances.py"
    smart_kf_script = repo_dir / "smart_keyframes_and_classify.py"
    assign_script = repo_dir / "assign_utterances_to_keyframes.py"
    build_demo_script = pipeline_dir / "build_final_output_demo_code.py"
    condense_script = repo_dir / "condense_final_output.py"

    for s in [deepgram_script, smart_kf_script, assign_script, build_demo_script, condense_script]:
        if not s.exists():
            raise FileNotFoundError(f"Script not found: {s}")

    utterances_json = out_dir / "utterances.json"
    keyframes_parsed_json = out_dir / "keyframes_parsed.json"
    keyframes_with_utterances_json = out_dir / "keyframes_with_utterances.json"
    final_output_json = out_dir / "final_output_demo_code.json"
    final_output_condensed_json = out_dir / "final_output_demo_code_condensed.json"
    deepgram_raw_json = Path(args.deepgram_raw_out).resolve() if args.deepgram_raw_out else None

    python_exe = str(Path(args.python))

    deepgram_cmd = [
        python_exe,
        str(deepgram_script),
        str(video_path),
        "-o",
        str(utterances_json),
        "--model",
        str(args.deepgram_model),
        "--request-timeout-sec",
        str(args.deepgram_request_timeout_sec),
        "--connect-timeout-sec",
        str(args.deepgram_connect_timeout_sec),
        "--retries",
        str(args.deepgram_retries),
        "--retry-backoff-sec",
        str(args.deepgram_retry_backoff_sec),
    ]
    if args.deepgram_language:
        deepgram_cmd.extend(["--language", str(args.deepgram_language)])
    if deepgram_raw_json is not None:
        deepgram_cmd.extend(["--raw", str(deepgram_raw_json)])

    smart_kf_cmd = [
        python_exe,
        str(smart_kf_script),
        "--video",
        str(video_path),
        "--out",
        str(out_dir),
        "--no-yolo-for-non-demo",
    ]
    if args.force_keyframes:
        smart_kf_cmd.append("--force")

    parallel_commands: List[Tuple[str, List[str]]] = []
    if args.force_deepgram or (not utterances_json.exists()):
        parallel_commands.append(("deepgram_extract_utterances", deepgram_cmd))
    else:
        print(f"[deepgram_extract_utterances] SKIP (exists): {utterances_json}")

    if args.force_keyframes or (not keyframes_parsed_json.exists()):
        parallel_commands.append(("smart_keyframes_and_classify", smart_kf_cmd))
    else:
        print(f"[smart_keyframes_and_classify] SKIP (exists): {keyframes_parsed_json}")

    if parallel_commands:
        print("Running Step 1+2 in parallel...")
        run_parallel(parallel_commands, cwd=repo_dir)
    else:
        print("Skipping Step 1+2 (all required artifacts already exist).")

    require_file(utterances_json, "deepgram_extract_utterances")
    require_file(keyframes_parsed_json, "smart_keyframes_and_classify")

    assign_cmd = [
        python_exe,
        str(assign_script),
        str(keyframes_parsed_json),
        str(utterances_json),
        "-o",
        str(keyframes_with_utterances_json),
        "--pre-roll-sec",
        str(args.pre_roll_sec),
    ]
    run_command("assign_utterances_to_keyframes", assign_cmd, cwd=repo_dir)
    require_file(keyframes_with_utterances_json, "assign_utterances_to_keyframes")

    build_cmd = [
        python_exe,
        str(build_demo_script),
        "--keyframes",
        str(keyframes_with_utterances_json),
        "--out",
        str(final_output_json),
        "--model",
        str(args.llm_model),
        "--similarity-threshold",
        str(args.similarity_threshold),
        "--temperature",
        str(args.temperature),
    ]
    run_command("build_final_output_demo_code", build_cmd, cwd=repo_dir)
    require_file(final_output_json, "build_final_output_demo_code")

    condense_cmd = [
        python_exe,
        str(condense_script),
        "--in",
        str(final_output_json),
        "--out",
        str(final_output_condensed_json),
    ]
    run_command("condense_final_output", condense_cmd, cwd=repo_dir)
    require_file(final_output_condensed_json, "condense_final_output")

    print("\nDemo-only Gemini pipeline completed successfully.")
    print(f"Utterances: {utterances_json}")
    print(f"Keyframes parsed: {keyframes_parsed_json}")
    print(f"Keyframes+utterances: {keyframes_with_utterances_json}")
    print(f"Final output (demo-only Gemini): {final_output_json}")
    print(f"Condensed output (demo-only Gemini): {final_output_condensed_json}")


if __name__ == "__main__":
    main()
