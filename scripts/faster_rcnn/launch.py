"""Submit one bounded pilot after baseline success; record all owned job IDs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
for name in ("evaluation", "islands", "evolution", "server"):
    (ROOT / "run_job_outputs" / name).mkdir(parents=True, exist_ok=True)
record = ROOT / "fred_faster_rcnn_output" / "launch.json"
record.parent.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--retry", action="store_true", help="Archive a finished launch and retry after repairs")
args = parser.parse_args()
if record.exists():
    old = json.loads(record.read_text())
    active = subprocess.check_output(["squeue", "--noheader", "--jobs=" + ",".join(old.values()), "--format=%i"], text=True).strip()
    if active or not args.retry:
        raise SystemExit(f"Launch record exists or jobs remain active: {record}")
    record.rename(record.with_name(f"launch-attempt-{old['baseline']}.json"))
env = dict(os.environ, LLMGE_DETECTOR="faster_rcnn")
submitted = {}

def submit(name, dependencies=(), server_start=None):
    command = ["sbatch", "--parsable", "--export=ALL,LLMGE_DETECTOR=faster_rcnn"]
    if dependencies:
        dependency = "afterok:" + ":".join(dependencies)
        if server_start:
            dependency += ",after:" + server_start
        command.append("--dependency=" + dependency)
        command.append("--kill-on-invalid-dep=yes")
    command.append(str(ROOT / "scripts/faster_rcnn" / f"{name}.sbatch"))
    value = subprocess.check_output(command, env=env, text=True).strip().split(";")[0]
    if not value.isdigit():
        raise RuntimeError(f"Unexpected Slurm job ID: {value}")
    submitted[name] = value
    record.write_text(json.dumps(submitted, indent=2))
    print(f"{name}: {value}", flush=True)
    return value

baseline = submit("baseline")
healthy = False
try:
    host = (ROOT / "fred_faster_rcnn_hostname.log").read_text().strip()
    with urllib.request.urlopen(f"http://{host}:{env.get('LLMGE_PORT', '8294')}/openapi.json", timeout=5) as response:
        healthy = response.status == 200
except (OSError, ValueError):
    pass
server_job = None
if not healthy:
    server_job = submit("server", (baseline,))
submit("pilot", (baseline,), server_start=server_job)
