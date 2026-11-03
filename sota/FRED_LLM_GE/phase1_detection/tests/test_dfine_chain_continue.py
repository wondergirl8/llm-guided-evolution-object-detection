"""Offline checks for D-FINE's Slurm continuation handoff."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[4] / "dfine_chain_continue.sbatch"
TAG = "dfine_20261102_eot0_chaincheck"


def run_transition(tmp_path, status, remaining="2"):
    root = tmp_path
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    (root / "run_job_outputs").mkdir()
    (root / "run_job_outputs/dfine_current_tag.txt").write_text(TAG + "\n")
    for script in ("server.sh", "run.sh"):
        (root / script).write_text("#!/bin/bash\n")
    run_root = root / "fred_runs" / TAG
    run_root.mkdir(parents=True)
    (run_root / "status.json").write_text(json.dumps(status))
    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "count=$(wc -l < \"$FAKE_SBATCH_LOG\")\n"
        "echo $((300 + count))\n"
    )
    sbatch.chmod(0o755)
    log = root / "sbatch.log"
    env = os.environ.copy()
    env.update(
        SLURM_SUBMIT_DIR=str(root),
        FRED_SEED_FAMILY="dfine",
        FRED_DFINE_SEARCH_SPACE="motifs_v1",
        FRED_NUM_GENERATIONS="21",
        FRED_RUN_TAG=TAG,
        FRED_CHAIN_ROOT=str(run_root / "chain"),
        SUBMIT_ISLAND_CONTROLLER="0",
        FAKE_SBATCH_LOG=str(log),
        PATH=f"{fake_bin}:{env['PATH']}",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "111", "222", remaining],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, log, run_root / "chain/jobs.tsv"


def test_paused_run_queues_server_controller_and_next_transition(tmp_path):
    result, log, jobs = run_transition(
        tmp_path, {"status": "paused_for_server_refresh", "server_job_id": "222"}
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    assert "--partition=coe-gpu" in calls[0]
    assert "server.sh" in calls[0]
    assert "--dependency=after:301+2" in calls[1]
    assert "LLMGE_SERVER_JOB_ID=301" in calls[1]
    assert "--dependency=afterok:302,afterany:301" in calls[2]
    assert jobs.read_text().splitlines() == [
        "server\t301\t111", "controller\t302\t301", "transition\t303\t302"
    ]


def test_complete_run_does_not_queue_another_window(tmp_path):
    result, log, _ = run_transition(
        tmp_path, {"status": "complete", "last_completed_generation": 20}
    )
    assert result.returncode == 0, result.stderr
    assert "no further jobs submitted" in result.stdout
    assert not log.exists()


def test_last_window_does_not_queue_another_transition(tmp_path):
    result, log, jobs = run_transition(
        tmp_path, {"status": "paused_for_server_refresh", "server_job_id": "222"},
        remaining="1",
    )
    assert result.returncode == 0, result.stderr
    assert len(log.read_text().splitlines()) == 2
    assert len(jobs.read_text().splitlines()) == 2


@pytest.mark.parametrize("status", [
    {"status": "running", "last_completed_generation": 3},
    {"status": "paused_for_server_refresh", "server_job_id": "999"},
    {"status": "complete", "last_completed_generation": 5},
])
def test_unexpected_status_never_queues_jobs(tmp_path, status):
    result, log, _ = run_transition(tmp_path, status)
    assert result.returncode != 0
    assert "Refusing to chain" in result.stderr
    assert not log.exists()
