#!/usr/bin/env bash
set -euo pipefail

export PORT="${PORT:-8080}"
export PARSE_WORKERS="${PARSE_WORKERS:-1}"
export GRADIO_CONCURRENCY="${GRADIO_CONCURRENCY:-1}"

python launch_lightning.py
