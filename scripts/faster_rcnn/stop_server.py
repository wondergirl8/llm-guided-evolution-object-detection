"""Release the dedicated server; never cancel a reused external server."""
import json
from pathlib import Path
import subprocess
record = json.loads(Path("fred_faster_rcnn_output/launch.json").read_text())
if "server" in record:
    subprocess.run(["scancel", str(record["server"])], check=False)
