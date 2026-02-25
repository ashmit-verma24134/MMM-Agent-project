# lightning-deployed-meet

Lightning AI-ready package for your meeting pipeline.

## What this package does
- `full` variant:
  - Gemini on all keyframe types.
- `demo-code` variant (your requested behavior):
  - Gemini + YOLO only for `demo` keyframes.
  - `slides`/`code`/`none` use OCR + transcript output.

## Lightning-ready defaults included
- Port defaults to `8080`.
- Auto GPU detection for runtime env:
  - `YOLO_DEVICE=0` and `OCR_MODE=gpu` when CUDA is available.
  - CPU fallbacks otherwise.
- Default parse pressure reduced for stability:
  - `PARSE_WORKERS=1`
  - `GRADIO_CONCURRENCY=1`
- Run artifacts default to persistent path in Lightning Studio when available:
  - `/teamspace/studios/this_studio/.cache/deployed-meet-runs`

## Quick start (Lightning Studio)
1. Open your Lightning Studio (GPU machine).
2. Upload or clone this folder into the Studio workspace.
3. Install dependencies:
```bash
cd lightning-deployed-meet
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
4. Set required keys:
```bash
export GEMINI_API_KEY="<your_key>"
export DEEPGRAM_API_KEY="<your_key>"
```
5. Start app:
```bash
python launch_lightning.py
```
6. Open the exposed app URL/port (`8080`) from Studio UI.

## Local run (Windows PowerShell)
```powershell
cd C:\meet-agent\lightning-deployed-meet
C:\meet-agent\.venv\Scripts\activate
pip install -r requirements.txt
$env:GEMINI_API_KEY="<your_key>"
$env:DEEPGRAM_API_KEY="<your_key>"
python launch_lightning.py
```

## Main files
- `launch_lightning.py`: Lightning-friendly launcher.
- `app.py`: Gradio UI.
- `run_manager.py`: run orchestration, run IDs, logs, outputs.
- `pipelines/`: pipeline scripts and model weights.
