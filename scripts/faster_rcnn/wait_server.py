"""Wait up to 20 minutes for the dedicated LLM server; never invent success."""
from pathlib import Path
import os
import time
import urllib.request
path = Path("fred_faster_rcnn_hostname.log")
for attempt in range(120):
    try:
        host = path.read_text().strip()
        with urllib.request.urlopen(f"http://{host}:{os.getenv('LLMGE_PORT', '8294')}/openapi.json", timeout=5) as response:
            if response.status == 200:
                print(f"LLM server ready on {host}", flush=True)
                break
    except (OSError, ValueError):
        time.sleep(10)
else:
    raise SystemExit("LLM server did not become ready within 20 minutes")
