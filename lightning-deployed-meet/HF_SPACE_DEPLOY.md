# Deploy to Hugging Face Spaces (Gradio SDK)

This package is now Gradio-native (`app.py`) and does not require Docker on Spaces.
The `demo-code` variant is configured as demo-only Gemini:
- Gemini + YOLO only for `demo` frames.
- `slides`/`code`/`none` use OCR + transcript output.

## 1) Create the Space
1. Go to `https://huggingface.co/new-space`.
2. Choose:
   - SDK: `Gradio`
   - Space name: your choice (for example `deployed-meet`)
   - Visibility: your choice
3. Click **Create Space**.

## 2) Clone the Space repo
```powershell
git clone https://huggingface.co/spaces/<YOUR_USER>/<YOUR_SPACE_NAME> hf-space-deployed-meet
cd hf-space-deployed-meet
```

## 3) Copy this folder into the Space repo
Copy everything from local `deployed-meet/` into this cloned Space repo root.

Required root files after copy:
- `app.py`
- `run_manager.py`
- `requirements.txt`
- `README.md`
- `pipelines/...`

## 4) Track model weights with Git LFS
```powershell
git lfs install
git lfs track "pipelines/models/*.pt"
git add .gitattributes
```

## 5) Add secrets in Space Settings
In **Settings -> Variables and secrets**, add:
- `GEMINI_API_KEY`
- `DEEPGRAM_API_KEY`

Optional:
- `PIPELINE_WORKDIR=/data/deployed-meet-runs`
- `YOLO_DEVICE=cpu` (if your Space has no GPU)
- `OCR_GPU=false` (if your Space has no GPU)

## 6) Commit and push
```powershell
git add .
git commit -m "Deploy deployed-meet Gradio app"
git push
```

Wait for the build to complete.

## 7) Open the app and run
- App URL: `https://<YOUR_SPACE_NAME>.hf.space`
- Start from **Start Run** tab, then monitor from **Track Run** tab.

## ZeroGPU note
- ZeroGPU only works with Gradio Spaces, which this repo now uses.
- This pipeline is long-running and model-heavy, so ZeroGPU sessions may be unstable for long videos.
- For reliable long jobs, CPU upgraded hardware or a dedicated GPU Space is recommended.
