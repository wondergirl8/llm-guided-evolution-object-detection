import json,subprocess
from pathlib import Path
path=Path("fred_faster_rcnn_scratch_output/launch.json")
if path.is_file():
    jobs=json.loads(path.read_text())
    if "server" in jobs: subprocess.run(["scancel",str(jobs["server"])],check=False)
