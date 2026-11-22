import json,subprocess,os
from pathlib import Path
path=Path("fred_runs")/os.environ["FRED_RUN_TAG"]/"launch.json"
if path.is_file():
    jobs=json.loads(path.read_text())
    if "server" in jobs: subprocess.run(["scancel",str(jobs["server"])],check=False)
