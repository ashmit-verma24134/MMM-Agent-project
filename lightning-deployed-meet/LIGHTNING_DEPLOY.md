# Deploy on Lightning AI (Step-by-step)

## 1) Create/open a GPU Lightning Studio
- Choose a machine with GPU time quota (your free GPU hours).
- Open a terminal inside Studio.

## 2) Get code into Studio
Option A: clone your repo
```bash
git clone <your_repo_url>
cd <your_repo>/lightning-deployed-meet
```

Option B: upload this folder directly, then:
```bash
cd lightning-deployed-meet
```

## 3) Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4) Configure required API keys
```bash
export GEMINI_API_KEY="<your_gemini_key>"
export DEEPGRAM_API_KEY="<your_deepgram_key>"
```

Optional tuning (recommended defaults are already in `launch_lightning.py`):
```bash
export PARSE_WORKERS=1
export GRADIO_CONCURRENCY=1
```

## 5) Start the app
```bash
python launch_lightning.py
```

App binds to `0.0.0.0:$PORT` (default `8080`).

## 6) Open app from Lightning UI
- In Studio, open the running service URL for port `8080`.
- Use **Start Run** tab to submit video.
- Use **Track Run** tab with `run_id` to watch logs and final output.

## 7) Storage notes
- Default run artifacts path:
  - `/teamspace/studios/this_studio/.cache/deployed-meet-runs`
- Override with:
```bash
export PIPELINE_WORKDIR="/teamspace/studios/this_studio/my_runs"
```

## 8) Common issues
- `CLIP dependencies missing`: ensure `requirements.txt` installed fully (includes `clip` and `torch`).
- Very long `artifact 2/5`: lower load by keeping:
  - `PARSE_WORKERS=1`
  - `GRADIO_CONCURRENCY=1`
- No GPU detected:
  - launcher falls back to CPU automatically (`YOLO_DEVICE=cpu`, `OCR_MODE=cpu`).
