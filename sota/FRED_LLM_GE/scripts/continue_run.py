"""Bounded Slurm renewal for an existing FRED checkpoint; dry-run unless --submit.

Uses the unchanged server.sh/run.sh. Each server gets a small CPU supervisor
dependent on server start; the supervisor waits for readiness and submits one
controller. It renews only at verified checkpoint boundaries after an expected
window exit, and never changes detector fitness, source, or checkpoints.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
SHARED = ("server.sh", "run.sh", "server.py", "run_improved.py")
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
ACTIVE_NAMES = {"LLMGE01_Server", "llm_opt", "llm_oper", "evaluateGene", "fred_renew"}
STOPPED_STATUSES = {"complete", "cycle_limit_reached", "stopped_on_error", "stopped_on_submission_error"}
MAX_SUPERVISOR_RESTARTS = 2


def command(argv, env=None, check=True):
    return subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, check=check)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seconds(value):
    """Accept Slurm's MM:SS, HH:MM:SS and D-HH:MM:SS forms."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?", value)
    if not match:
        raise ValueError(f"Unrecognized Slurm duration: {value!r}")
    days, first, second, third = match.groups()
    if third is None:
        if days is not None or int(second) >= 60:
            raise ValueError(f"Invalid Slurm duration: {value!r}")
        return int(first) * 60 + int(second)
    if int(second) >= 60 or int(third) >= 60:
        raise ValueError(f"Invalid Slurm duration: {value!r}")
    return int(days or 0) * 86400 + int(first) * 3600 + int(second) * 60 + int(third)


def job(job_id):
    queued = command(["squeue", "-h", "-j", str(job_id), "-o", "%T|%B|%L"], check=False)
    if queued.stdout.strip():
        state, host, remaining = queued.stdout.strip().split("|")
        return dict(state=state, host=host, remaining=remaining)
    if queued.returncode and "Invalid job id" not in queued.stderr:
        raise RuntimeError(queued.stderr.strip())
    accounted = command(["sacct", "-nP", "-j", str(job_id), "--format=JobID,State,ExitCode"])
    for line in accounted.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == str(job_id):
            return dict(state=fields[1].split()[0].rstrip("+"), exit_code=fields[2])
    # Accounting can briefly lag squeue; this state is never treated as success.
    return dict(state="ACCOUNTING_PENDING")


def queued_jobs():
    result = command(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j"])
    return [line.strip().split("|", 1) for line in result.stdout.splitlines() if line.strip()]


def assert_no_other_run():
    active = []
    for identifier, name in queued_jobs():
        if name not in ACTIVE_NAMES:
            continue
        details = command(["scontrol", "show", "job", "-o", identifier], check=False)
        if details.returncode:
            raise RuntimeError(f"Cannot verify existing job {identifier}: {details.stderr}")
        workdir = re.search(r"\bWorkDir=(\S+)", details.stdout)
        script = re.search(r"\bCommand=(\S+)", details.stdout)
        if ((workdir and Path(workdir[1]).resolve() == ROOT.resolve()) or
                (script and Path(script[1]).resolve().is_relative_to(ROOT.resolve()))):
            active.append((identifier, name))
    if active:
        raise RuntimeError(f"Existing server/controller/operator/evaluation jobs: {active}; do not overlap runs")


def snapshot(env):
    """Use the real loader and fingerprint, without executing the evolution loop."""
    code = """import json
import run_improved as e
p,n,g=e.load_checkpoint(folder_name=e.RUN_CHECKPOINT_DIR,global_path=e.RUN_CHECKPOINT_DIR)
if not p or n < 1: raise RuntimeError('A completed checkpoint pair is required')
print(json.dumps(dict(generation=n-1, next_generation=n, population=len(p['population']),
                     fingerprint=g['protocol_fingerprint'])))
"""
    result = command([str(ROOT / ".venv/bin/python"), "-c", code], env=env)
    return json.loads(result.stdout.splitlines()[-1])


def environment(config):
    env = dict(os.environ, **config["environment"])
    env.pop("FRED_EVENTS", None)
    return env


def verify_inputs(config):
    if config["shared_hashes"] != {name: digest(ROOT / name) for name in SHARED}:
        raise RuntimeError("Shared execution code changed; refusing automatic continuation")
    if digest(SCRIPT) != config["supervisor_sha256"]:
        raise RuntimeError("Continuation supervisor changed; refusing automatic continuation")
    current = snapshot(environment(config))
    if current["fingerprint"] != config["fingerprint"]:
        raise RuntimeError("Run fingerprint changed; refusing automatic continuation")
    return current


@contextmanager
def lock():
    # hostname.log is shared by all families, so serialize renewal across this checkout.
    path = ROOT / "run_job_outputs/fred_renew.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another continuation supervisor is active") from None
        yield


def save(path, config, status):
    config.update(status=status, updated_at=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)
    print(status, flush=True)


def submit(argv, env):
    # A child supervisor has its own restart counter; do not inherit its parent's.
    env = dict(env)
    env.pop("SLURM_RESTART_COUNT", None)
    value = command(["sbatch", "--parsable", "--export=ALL", *argv], env=env).stdout.strip().split(";")[0]
    if not value.isdecimal():
        raise RuntimeError(f"Unexpected sbatch response: {value!r}")
    return value


def cancel_owned(job_id):
    if job_id and job(job_id)["state"] not in TERMINAL:
        command(["scancel", str(job_id)])


def wait_until(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(30)
    raise RuntimeError(f"Timed out waiting for {description}")


def submit_cycle(path, config):
    current = verify_inputs(config)
    env = environment(config)
    cycle = dict(number=len(config["cycles"]) + 1, start_generation=current["generation"],
                 server=None, controller=None, supervisor=None)
    config["cycles"].append(cycle)
    save(path, config, "submitting_server")
    try:
        cycle["server"] = submit([str(ROOT / "server.sh")], env)
        save(path, config, "submitting_supervisor")
        wrapped = shlex.join([str(ROOT / ".venv/bin/python"), str(SCRIPT),
                              "supervise", "--state", str(path), "--cycle", str(cycle["number"])])
        cycle["supervisor"] = submit([
            "--job-name=fred_renew", "--partition=pace-cpu", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=1", "--mem=4G", "--time=18:00:00", "--requeue", "--open-mode=append",
            f"--dependency=after:{cycle['server']}", "--kill-on-invalid-dep=yes",
            f"--output={path.parent}/supervisor-%j.out", "--wrap", wrapped], env)
        save(path, config, "waiting_for_server_allocation")
    except BaseException:
        for identifier in (cycle["supervisor"], cycle["server"]):
            if identifier:
                cancel_owned(identifier)
        raise


def ready(server_id, port):
    current = job(server_id)
    if current["state"] in TERMINAL:
        raise RuntimeError(f"Server {server_id} ended before readiness: {current}")
    if current["state"] != "RUNNING":
        return False
    if seconds(current["remaining"]) < 3600:
        raise RuntimeError("Server has less than one hour left before controller startup")
    log = ROOT / f"run_job_outputs/server/slurm-{server_id}.out"
    if not log.exists() or "Application startup complete." not in log.read_text(errors="replace"):
        return False
    # Use this job's batch host, not a possibly stale global hostname file.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{current['host']}:{port}/", timeout=5) as response:
            if response.status != 200:
                return False
    except (OSError, urllib.error.URLError):
        return False
    published = (ROOT / "hostname.log").read_text().strip()
    if published.split(".")[0] != current["host"].split(".")[0]:
        raise RuntimeError("hostname.log does not identify this server; another run may be active")
    return True


def wait_controller(cycle):
    def finished():
        controller = job(cycle["controller"])
        if controller["state"] in TERMINAL:
            return controller
        server = job(cycle["server"])
        if server["state"] in TERMINAL and server["state"] != "TIMEOUT":
            raise RuntimeError(f"Server exited unexpectedly while controller is active: {server}")
        return False
    return wait_until(finished, 10 * 3600, "controller completion (including queue time)")


def restart_reason(controller, server, log, progressed, server_id=None):
    """Classify narrowly; preserve unknown errors rather than restart forever."""
    if not progressed:
        raise RuntimeError("No new generation checkpoint; refusing an automatic retry loop")
    if server["state"] in TERMINAL and server["state"] != "TIMEOUT":
        raise RuntimeError(f"Unexpected server outcome: {server}")
    state = controller["state"]
    if state == "COMPLETED" and "-- Paused for server refresh --" in log:
        return "clean_server_window_pause"
    if state == "TIMEOUT":
        return "controller_walltime_after_checkpoint_progress"
    if state == "FAILED":
        exceptions = re.findall(r"^(?:[\w.]*Error|[\w.]*Exception|ServerWindowClosed): .*?$", log, re.MULTILINE)
        final_error = exceptions[-1] if exceptions else ""
        match = re.fullmatch(r"RuntimeError: Unrecognized server time remaining from squeue: '([0-9:]+)'", final_error)
        if match and match[1].count(":") == 1 and seconds(match[1]) < 3600:
            return "known_MM_SS_server_window_parser_error"
        if final_error.startswith("ServerWindowClosed:") and "require at least" in final_error:
            return "server_window_closed_during_operator_submission"
        if server["state"] == "TIMEOUT" and server_id:
            if final_error == f"ServerWindowClosed: LLM server job {server_id} is no longer in squeue":
                return "server_expired_after_checkpoint_progress"
            prefix = f"RuntimeError: Cannot check LLM server job {server_id}: "
            if final_error.startswith(prefix) and "Invalid job id" in final_error[len(prefix):]:
                return "expired_server_removed_from_squeue"
    raise RuntimeError(f"Unexpected controller outcome {controller}; inspect its original log")


def drain_children(config):
    operators = (Path(config["run_root"]) / "operator_jobs").resolve()
    def drained():
        pending = []
        for identifier, name in queued_jobs():
            if name not in ("llm_oper", "evaluateGene"):
                continue
            details = command(["scontrol", "show", "job", "-o", identifier], check=False)
            if details.returncode:
                raise RuntimeError(f"Cannot verify child job {identifier}: {details.stderr}")
            match = re.search(r"\bCommand=(\S+)", details.stdout)
            if not match:
                raise RuntimeError(f"Missing command for child job {identifier}")
            if Path(match[1]).resolve().is_relative_to(operators):
                pending.append(identifier)
        if pending:
            print(f"Waiting for existing child jobs: {pending}", flush=True)
        return not pending
    wait_until(drained, 5 * 3600, "existing run child jobs to finish")


def supervise(path, number):
    with lock():
        config = json.loads(path.read_text())
        if number < len(config["cycles"]):
            latest = config["cycles"][-1]
            if latest.get("supervisor") is None:
                raise RuntimeError("Interrupted next-cycle submission; inspect state and Slurm before restarting")
            print("This cycle already handed over to its successor; nothing submitted.", flush=True)
            return
        if number != len(config["cycles"]):
            raise RuntimeError("Stale supervisor cycle")
        if config["status"] in STOPPED_STATUSES:
            print(f"Chain already stopped: {config['status']}; nothing submitted.", flush=True)
            return
        cycle = config["cycles"][-1]
        try:
            verify_inputs(config)
            restarts = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
            if restarts < 0 or restarts > MAX_SUPERVISOR_RESTARTS:
                raise RuntimeError(f"Supervisor restart limit exceeded: {restarts}; inspect the cluster failure")
            cycle["supervisor_restarts"] = restarts
            if cycle["controller"] is None:
                if config["status"] not in {"waiting_for_server_allocation", "waiting_for_model_readiness"}:
                    raise RuntimeError("Interrupted controller submission; inspect Slurm for an unrecorded controller")
                save(path, config, "waiting_for_model_readiness")
                wait_until(lambda: ready(cycle["server"], config["port"]), 3600, "model readiness")
                verify_inputs(config)
                env = environment(config)
                env["LLMGE_SERVER_JOB_ID"] = cycle["server"]
                # If killed between sbatch and recording its ID, stop for review on
                # re-entry rather than risk submitting a duplicate controller.
                save(path, config, "submitting_controller")
                cycle["controller"] = submit([str(ROOT / "run.sh")], env)
                save(path, config, "controller_running_or_queued")
            else:
                print(f"Resuming supervision of controller {cycle['controller']}.", flush=True)
                save(path, config, config["status"])
            outcome = wait_controller(cycle)
            cycle["controller_outcome"] = outcome
            current = verify_inputs(config)
            cycle["last_completed_generation"] = current["generation"]
            log = (ROOT / f"run_job_outputs/islands/slurm-{cycle['controller']}.out").read_text(errors="replace")
            if current["generation"] >= config["generations"] - 1:
                if outcome["state"] != "COMPLETED" or "-- End of Era --" not in log:
                    raise RuntimeError("Final checkpoint exists but controller did not report clean completion")
                cancel_owned(cycle["server"])
                save(path, config, "complete")
                return
            if config["status"] in {"releasing_previous_server", "draining_previous_children"}:
                if not cycle.get("renewal_reason") or current["generation"] <= cycle["start_generation"]:
                    raise RuntimeError("Missing verified checkpoint progress for interrupted renewal")
            else:
                cycle["renewal_reason"] = restart_reason(
                    outcome, job(cycle["server"]), log, current["generation"] > cycle["start_generation"],
                    server_id=cycle["server"])
                # Persist the approved release before cancelling our server so a
                # restarted supervisor can distinguish it from an unexpected exit.
                save(path, config, "releasing_previous_server")
            # Release only the server owned by this cycle; never cancel model evaluations.
            cancel_owned(cycle["server"])
            wait_until(lambda: job(cycle["server"])["state"] in TERMINAL, 300, "old server release")
            save(path, config, "draining_previous_children")
            drain_children(config)
            if len(config["cycles"]) >= config["max_cycles"]:
                save(path, config, "cycle_limit_reached")
                return
            submit_cycle(path, config)
        except BaseException as exc:
            config["error"] = str(exc)
            # These IDs were created by this chain; unrelated jobs are never cancelled.
            cleanup_errors = []
            for identifier in (cycle.get("controller"), cycle.get("server")):
                try:
                    cancel_owned(identifier)
                except Exception as cleanup:
                    cleanup_errors.append(str(cleanup))
            if cleanup_errors:
                config["cleanup_errors"] = cleanup_errors
            save(path, config, "stopped_on_error")
            raise


def start(args):
    if args.generations < 1 or args.cycles < 1 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid generation count, cycle limit, or port")
    if not re.fullmatch(rf"{re.escape(args.family)}_[0-9]{{8}}_eot0_[a-z0-9][a-z0-9_-]*", args.run_tag):
        raise ValueError("Use the existing family_YYYYMMDD_eot0_name run tag")
    env_values = {key: os.environ[key] for key in (
        "LLMGE_MODEL_PATH", "LLMGE_LLM_MODEL", "LLM_MAX_NEW_TOKENS", "LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS",
        "FRED_EVAL_GPU_CONSTRAINT", "LLMGE_MIN_SERVER_REMAINING_SECONDS") if key in os.environ}
    env_values.update(FRED_SEED_FAMILY=args.family, FRED_RUN_TAG=args.run_tag,
                      FRED_NUM_GENERATIONS=str(args.generations), LLMGE_PORT=str(args.port),
                      SUBMIT_ISLAND_CONTROLLER="0", UV_NO_SYNC="1", LLMGE_AUTO_START_SERVER="0")
    config = dict(schema_version="fred-continuation-chain-v1", run_tag=args.run_tag,
                  run_root=str(ROOT / "fred_runs" / args.run_tag), generations=args.generations,
                  max_cycles=args.cycles, port=args.port, environment=env_values, cycles=[],
                  shared_hashes={name: digest(ROOT / name) for name in SHARED},
                  supervisor_sha256=digest(SCRIPT))
    current = snapshot(environment(config))
    config["fingerprint"] = current["fingerprint"]
    print(json.dumps(dict(next_generation=current["next_generation"], population=current["population"],
                          maximum_server_windows=args.cycles, server_hours_per_window=8,
                          shared_launch_files="unchanged", submit=args.submit), indent=2))
    if current["generation"] >= args.generations - 1:
        print("Run already reached its final checkpoint; nothing submitted.")
        return
    if not args.submit:
        print("Dry run only. Add --submit to enqueue the bounded continuation chain.")
        return
    with lock():
        assert_no_other_run()
        directory = Path(config["run_root"]) / "continuation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory.mkdir(parents=True, exist_ok=False)
        try:
            submit_cycle(directory / "state.json", config)
        except BaseException as exc:
            config["error"] = str(exc)
            save(directory / "state.json", config, "stopped_on_submission_error")
            raise
        print(f"Chain state: {directory / 'state.json'}")


def show_status(run_tag):
    if not re.fullmatch(r"[a-z0-9_]+", run_tag):
        raise ValueError("Invalid run tag")
    paths = sorted((ROOT / "fred_runs" / run_tag / "continuation").glob("*/state.json"))
    if not paths:
        print("No continuation chain has been submitted for this tag.")
        return
    config = json.loads(paths[-1].read_text())
    jobs = {}
    if config["cycles"]:
        cycle = config["cycles"][-1]
        for role in ("server", "controller", "supervisor"):
            if cycle.get(role):
                jobs[role] = dict(job_id=cycle[role], **job(cycle[role]))
    status = config["status"]
    if status not in STOPPED_STATUSES and jobs.get("supervisor", {}).get("state") in TERMINAL:
        status = "supervisor_stopped"
    print(json.dumps(dict(state_file=str(paths[-1]), status=status, recorded_status=config["status"], jobs=jobs,
                          maximum_server_windows=config["max_cycles"],
                          cycles=config["cycles"], error=config.get("error")), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("start")
    plan.add_argument("--family", required=True)
    plan.add_argument("--run-tag", required=True)
    plan.add_argument("--generations", type=int, required=True)
    plan.add_argument("--cycles", type=int, default=3)
    plan.add_argument("--port", type=int, default=8479)
    plan.add_argument("--submit", action="store_true")
    worker = commands.add_parser("supervise")
    worker.add_argument("--state", type=Path, required=True)
    worker.add_argument("--cycle", type=int, required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    if args.command == "start":
        start(args)
    elif args.command == "supervise":
        supervise(args.state.resolve(), args.cycle)
    else:
        show_status(args.run_tag)


if __name__ == "__main__":
    main()
