#!/usr/bin/env python3
"""
deepgram_extract_utterances.py

Extract speaker-attributed utterances (start, end, speaker, text)
from a meeting MP4 using Deepgram.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource


# load .env at startup
load_dotenv()


def _die(msg: str, code: int = 1) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def _load_file_source(path: str):
    if not os.path.isfile(path):
        _die(f"File not found: {path}")

    with open(path, "rb") as f:
        data = f.read()

    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "application/octet-stream"

    # IMPORTANT: return a dict, NOT FileSource()
    return {
        "buffer": data,
        "mimetype": mime,
    }



def _extract_utterances(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    utterances = result.get("results", {}).get("utterances", [])
    out: List[Dict[str, Any]] = []

    for u in utterances:
        out.append(
            {
                "start": float(u.get("start", 0.0)),
                "end": float(u.get("end", 0.0)),
                "speaker": u.get("speaker"),
                "text": (u.get("transcript") or "").strip(),
            }
        )

    return out


def _is_non_retryable_error(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and 400 <= code < 500:
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, int) and 400 <= status < 500:
        return True
    msg = str(exc).lower()
    # Deepgram SDK exceptions often encode status in message text.
    if "status: 4" in msg or "bad request" in msg or "unsupported data" in msg:
        return True
    return False


def transcribe_and_extract(
    path: str,
    model: str = "nova-3",
    language: Optional[str] = None,
    request_timeout_sec: float = 1200.0,
    connect_timeout_sec: float = 30.0,
    retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        _die("DEEPGRAM_API_KEY not found in environment or .env")

    client = DeepgramClient(api_key=api_key)

    source = _load_file_source(path)

    options_kwargs: Dict[str, Any] = {
        "model": model,
        "smart_format": True,
        "punctuate": True,
        "utterances": True,
        "diarize": True,
    }
    if language:
        options_kwargs["language"] = language

    options = PrerecordedOptions(**options_kwargs)

    # Deepgram SDK default HTTP timeout is 30s; long recordings often exceed that.
    timeout = httpx.Timeout(float(request_timeout_sec), connect=float(connect_timeout_sec))
    retries = max(1, int(retries))

    last_err: Optional[Exception] = None
    response = None
    for attempt in range(1, retries + 1):
        try:
            response = client.listen.rest.v("1").transcribe_file(
                source,
                options,
                timeout=timeout,
            )
            break
        except Exception as e:
            last_err = e
            if _is_non_retryable_error(e):
                # Client/input errors won't succeed on retry.
                raise
            if attempt >= retries:
                raise
            wait_sec = float(retry_backoff_sec) * attempt
            print(
                f"Deepgram request failed (attempt {attempt}/{retries}): {type(e).__name__}: {e}. "
                f"Retrying in {wait_sec:.1f}s..."
            )
            time.sleep(wait_sec)

    if response is None:
        raise RuntimeError(f"Deepgram transcription failed after {retries} attempts: {last_err}")

    result_dict = response.to_dict() if hasattr(response, "to_dict") else dict(response)

    return {
        "input_file": os.path.abspath(path),
        "model": model,
        "utterances": _extract_utterances(result_dict),
    }, result_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to meeting file (.mp4, .wav, .mp3)")
    parser.add_argument("-o", "--output", default="utterances.json")
    parser.add_argument("--raw", help="Optional raw Deepgram response JSON")
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--language", help="Optional language code (e.g. en, en-US)")
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=1200.0,
        help="HTTP request timeout for Deepgram API call (default: 1200s).",
    )
    parser.add_argument(
        "--connect-timeout-sec",
        type=float,
        default=30.0,
        help="HTTP connect timeout for Deepgram API call (default: 30s).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts for Deepgram call (default: 3).",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=2.0,
        help="Base retry backoff seconds; actual sleep is base * attempt (default: 2.0).",
    )
    args = parser.parse_args()

    extracted, raw = transcribe_and_extract(
        args.input,
        model=args.model,
        language=args.language,
        request_timeout_sec=float(args.request_timeout_sec),
        connect_timeout_sec=float(args.connect_timeout_sec),
        retries=int(args.retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    if args.raw:
        with open(args.raw, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f"Saved utterances to {args.output}")
    if args.raw:
        print(f"Saved raw response to {args.raw}")


if __name__ == "__main__":
    main()
