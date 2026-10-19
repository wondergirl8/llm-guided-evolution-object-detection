"""Submit or resume the scratch workflow; this file never runs on import."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    os.chdir(ROOT)
    for name in ("evaluation","islands","evolution","server"): (ROOT/"run_job_outputs"/name).mkdir(parents=True,exist_ok=True)
    output=ROOT/"fred_faster_rcnn_scratch_output"; output.mkdir(parents=True,exist_ok=True); record=output/"launch.json"
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--resume",action="store_true"); args=p.parse_args(); submitted={}
    if record.exists():
        old=json.loads(record.read_text()); ids=[str(v) for v in old.values() if str(v).isdigit()]
        active=subprocess.check_output(["squeue","--noheader","--jobs="+",".join(ids),"--format=%i"],text=True).strip() if ids else ""
        if active and set(old)!={"baseline","server","evolve","finalize","report"}: submitted.update(old)
        elif active: raise SystemExit(f"Scratch launch still has active jobs: {active}")
        else:
            if not args.resume: raise SystemExit("Launch record exists; use --resume")
            record.rename(output/f"launch-{int(time.time())}.json")
    env=dict(os.environ,LLMGE_DETECTOR="faster_rcnn_scratch",LLMGE_PORT="8295")
    def submit(name,dependency=None):
        command=["sbatch","--parsable","--export=ALL,LLMGE_DETECTOR=faster_rcnn_scratch,LLMGE_PORT=8295"]
        if dependency: command += ["--dependency="+dependency,"--kill-on-invalid-dep=yes"]
        command.append(str(ROOT/"scripts/faster_rcnn_scratch"/f"{name}.sbatch")); job=subprocess.check_output(command,env=env,text=True).strip().split(";")[0]
        if not job.isdigit(): raise RuntimeError(f"Unexpected Slurm job ID: {job}")
        submitted[name]=job; record.write_text(json.dumps(submitted,indent=2)+"\n"); print(f"{name}: {job}"); return job
    gen20=ROOT/"fred_faster_rcnn_scratch_checkpoints/global_gen_20.pkl"; held=ROOT/"sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/held_out_reports"
    if not gen20.exists():
        baseline=submitted.get("baseline"); seed=ROOT/"sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/results/seed_results.csv"
        if not baseline and not seed.exists(): baseline=submit("baseline")
        server=submitted.get("server") or submit("server",f"afterok:{baseline}" if baseline else None)
        deps=([f"afterok:{baseline}"] if baseline else [])+[f"after:{server}"]
        evolution=submitted.get("evolve") or submit("evolve",",".join(deps)); final=submitted.get("finalize") or submit("finalize",f"afterok:{evolution}")
        if "report" not in submitted: submit("report",f"afterok:{final}")
    elif not (held.is_dir() and any(held.glob("*_evaluation.json"))):
        final=submit("finalize"); submit("report",f"afterok:{final}")
    else: submit("report")
if __name__=="__main__": main()
