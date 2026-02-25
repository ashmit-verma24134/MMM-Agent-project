$env:PORT = if ($env:PORT) { $env:PORT } else { "8080" }
$env:PARSE_WORKERS = if ($env:PARSE_WORKERS) { $env:PARSE_WORKERS } else { "1" }
$env:GRADIO_CONCURRENCY = if ($env:GRADIO_CONCURRENCY) { $env:GRADIO_CONCURRENCY } else { "1" }

python launch_lightning.py
