from pathlib import Path
import os,time,urllib.request
path=Path("fred_faster_rcnn_scratch_hostname.log")
for _ in range(360):
    try:
        host=path.read_text().strip()
        if urllib.request.urlopen(f"http://{host}:{os.getenv('LLMGE_PORT','8295')}/openapi.json",timeout=5).status==200: break
    except (OSError,ValueError): time.sleep(10)
else: raise SystemExit("LLM server did not become ready within 60 minutes")
