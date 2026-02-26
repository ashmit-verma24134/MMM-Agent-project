from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import gradio as gr

from run_manager import get_final_output, get_logs, get_status, start_run


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _err_payload(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def start_pipeline(
    variant: str,
    input_mode: str,
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
    llm_model: str,
    similarity_threshold: float,
    temperature: float,
    log_heartbeat_sec: float,
) -> Tuple[str, Dict[str, Any], str, str]:
    try:
        chosen_video_file = None
        chosen_video_url = None
        mode = (input_mode or "").strip().lower()

        if mode == "upload file":
            chosen_video_file = _clean_optional(video_file_path)
            if not chosen_video_file:
                raise ValueError("Select a video file for Upload File mode.")
        elif mode == "video url":
            chosen_video_url = _clean_optional(video_url)
            if not chosen_video_url:
                raise ValueError("Provide video_url for Video URL mode.")
        else:
            raise ValueError("Invalid input mode.")

        result = start_run(
            variant=variant,
            video_file_path=chosen_video_file,
            video_url=chosen_video_url,
            out_dir=_clean_optional(out_dir),
            python_bin=_clean_optional(python_bin),
            deepgram_model=deepgram_model,
            deepgram_language=_clean_optional(deepgram_language),
            deepgram_request_timeout_sec=float(deepgram_request_timeout_sec),
            deepgram_connect_timeout_sec=float(deepgram_connect_timeout_sec),
            deepgram_retries=int(deepgram_retries),
            deepgram_retry_backoff_sec=float(deepgram_retry_backoff_sec),
            force_deepgram=bool(force_deepgram),
            force_keyframes=bool(force_keyframes),
            pre_roll_sec=float(pre_roll_sec),
            llm_model=llm_model,
            similarity_threshold=float(similarity_threshold),
            temperature=float(temperature),
            log_heartbeat_sec=float(log_heartbeat_sec),
        )
        run_id = str(result["run_id"])
        logs = get_logs(run_id, tail_lines=120)
        return run_id, result, logs, run_id
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        return "", _err_payload(msg), msg, ""


def refresh_status_logs(run_id: str, tail_lines: int) -> Tuple[Dict[str, Any], str]:
    rid = _clean_optional(run_id)
    if not rid:
        return _err_payload("Enter a run_id."), ""
    try:
        status = get_status(rid)
        logs = get_logs(rid, tail_lines=int(tail_lines))
        return status, logs
    except Exception as e:
        return _err_payload(f"{type(e).__name__}: {e}"), ""


def fetch_output(run_id: str, condensed: bool) -> Dict[str, Any]:
    rid = _clean_optional(run_id)
    if not rid:
        return _err_payload("Enter a run_id.")
    try:
        return get_final_output(rid, condensed=condensed)
    except Exception as e:
        return _err_payload(f"{type(e).__name__}: {e}")


def watch_run(
    run_id: str,
    tail_lines: int,
    poll_sec: float,
):
    rid = _clean_optional(run_id)
    if not rid:
        yield _err_payload("Enter a run_id."), "", None, None
        return

    sleep_sec = max(1.0, float(poll_sec))
    max_tail = max(10, min(int(tail_lines), 5000))

    while True:
        try:
            status = get_status(rid)
            logs = get_logs(rid, tail_lines=max_tail)
        except Exception as e:
            yield _err_payload(f"{type(e).__name__}: {e}"), "", None, None
            return

        state = str(status.get("status", "unknown")).lower()
        if state in {"succeeded", "failed"}:
            full_payload = None
            condensed_payload = None
            if state == "succeeded":
                try:
                    full_payload = get_final_output(rid, condensed=False)
                except Exception as e:
                    full_payload = _err_payload(f"{type(e).__name__}: {e}")
                try:
                    condensed_payload = get_final_output(rid, condensed=True)
                except Exception as e:
                    condensed_payload = _err_payload(f"{type(e).__name__}: {e}")
            yield status, logs, full_payload, condensed_payload
            return

        yield status, logs, None, None
        time.sleep(sleep_sec)


with gr.Blocks(title="deployed-meet") as demo:
    gr.Markdown(
        """
        # deployed-meet (Gradio)
        Start either pipeline variant, then monitor logs and fetch final outputs by `run_id`.
        - `full`: Gemini on all keyframe types.
        - `demo-code`: Gemini only on demo keyframes, slides+code are OCR/transcript based.
        """
    )

    with gr.Tab("Start Run"):
        variant = gr.Dropdown(
            choices=[
                ("Full pipeline (Gemini on slides/code/demo)", "full"),
                ("Demo-only Gemini pipeline (slides+code OCR)", "demo-code"),
            ],
            value="demo-code",
            label="Pipeline Variant",
        )
        input_mode = gr.Radio(
            choices=["Upload File", "Video URL"],
            value="Upload File",
            label="Input Mode",
        )
        video_file = gr.File(label="Video File", type="filepath")
        video_url = gr.Textbox(label="Video URL", placeholder="https://.../meeting.mp4")

        out_dir = gr.Textbox(
            label="Output Directory (optional)",
            placeholder="run_001",
        )
        python_bin = gr.Textbox(
            label="Python Executable (optional)",
            placeholder="Leave blank to auto-resolve",
        )

        with gr.Accordion("Advanced Settings", open=False):
            deepgram_model = gr.Textbox(label="Deepgram Model", value="nova-3")
            deepgram_language = gr.Textbox(label="Deepgram Language (optional)", value="")
            deepgram_request_timeout_sec = gr.Number(label="Deepgram Request Timeout (sec)", value=1200.0)
            deepgram_connect_timeout_sec = gr.Number(label="Deepgram Connect Timeout (sec)", value=30.0)
            deepgram_retries = gr.Number(label="Deepgram Retries", value=3, precision=0)
            deepgram_retry_backoff_sec = gr.Number(label="Deepgram Retry Backoff (sec)", value=2.0)
            force_deepgram = gr.Checkbox(label="Force Deepgram Re-run", value=False)
            force_keyframes = gr.Checkbox(label="Force Keyframe Re-run", value=False)
            pre_roll_sec = gr.Number(label="Pre-roll Seconds", value=3.0)
            llm_model = gr.Textbox(label="LLM Model", value="llama-3.3-70b-versatile")
            similarity_threshold = gr.Number(label="Similarity Threshold", value=0.82)
            temperature = gr.Number(label="Temperature", value=0.2)
            log_heartbeat_sec = gr.Number(label="Heartbeat Log Interval (sec)", value=10.0)

        start_btn = gr.Button("Start Pipeline", variant="primary")
        start_run_id = gr.Textbox(label="Run ID", interactive=False)
        start_status = gr.JSON(label="Start Response / Error")
        start_logs = gr.Textbox(label="Initial Logs", lines=14)

    with gr.Tab("Track Run"):
        track_run_id = gr.Textbox(label="Run ID", placeholder="Paste run_id from Start tab")
        tail_lines = gr.Slider(label="Log Tail Lines", minimum=50, maximum=3000, value=300, step=50)
        poll_sec = gr.Slider(label="Live Poll Interval (sec)", minimum=1, maximum=20, value=3, step=1)

        with gr.Row():
            refresh_btn = gr.Button("Refresh Status + Logs")
            watch_btn = gr.Button("Watch Live")
            full_btn = gr.Button("Fetch Final Output")
            condensed_btn = gr.Button("Fetch Condensed Output")

        track_status = gr.JSON(label="Run Status")
        track_logs = gr.Textbox(label="Run Logs", lines=22)
        track_full_output = gr.JSON(label="Final Output")
        track_condensed_output = gr.JSON(label="Condensed Final Output")

    start_btn.click(
        fn=start_pipeline,
        inputs=[
            variant,
            input_mode,
            video_file,
            video_url,
            out_dir,
            python_bin,
            deepgram_model,
            deepgram_language,
            deepgram_request_timeout_sec,
            deepgram_connect_timeout_sec,
            deepgram_retries,
            deepgram_retry_backoff_sec,
            force_deepgram,
            force_keyframes,
            pre_roll_sec,
            llm_model,
            similarity_threshold,
            temperature,
            log_heartbeat_sec,
        ],
        outputs=[start_run_id, start_status, start_logs, track_run_id],
    )

    refresh_btn.click(
        fn=refresh_status_logs,
        inputs=[track_run_id, tail_lines],
        outputs=[track_status, track_logs],
    )

    watch_btn.click(
        fn=watch_run,
        inputs=[track_run_id, tail_lines, poll_sec],
        outputs=[track_status, track_logs, track_full_output, track_condensed_output],
    )

    full_btn.click(
        fn=lambda rid: fetch_output(rid, False),
        inputs=[track_run_id],
        outputs=[track_full_output],
    )

    condensed_btn.click(
        fn=lambda rid: fetch_output(rid, True),
        inputs=[track_run_id],
        outputs=[track_condensed_output],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=int(os.getenv("GRADIO_CONCURRENCY", "2"))).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "8080")),
        show_error=True,
    )
