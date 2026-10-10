"""Prepare one validated representative per scene, without editing old artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sota.FRED_LLM_GE.archive.phase0_data_old.annotation_policy import POLICY_VERSION
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import load_project_split_manifest
from sota.FRED_LLM_GE.phase0_data.check_scene_split import DEFAULT_PROJECT, main as check_scene_split
from sota.FRED_LLM_GE.data.scene_diagnostic_export import FRAMES_PER_SEQUENCE, export, verify_export


def prepare(directory: Path) -> None:
    check_scene_split([])
    project = load_project_split_manifest(DEFAULT_PROJECT)
    manifest = directory / "manifest.sqlite"
    if manifest.exists() or (directory / "export").exists():
        raise FileExistsError("refusing to reuse diagnostic outputs")
    inspections = directory / "inspections"
    inspections.mkdir()
    inputs = ["--inventory", "data/fred_phase0/metadata/source_inventory.json",
              "--official-split", "data/fred_phase0/metadata/official_challenging_split.json",
              "--project-split", str(DEFAULT_PROJECT), "--annotation-policy", POLICY_VERSION]
    command = [sys.executable, "-m", "sota.FRED_LLM_GE.archive.phase0_data_old"]
    selected = {}
    for name, members in project.recording_groups:
        candidates = sorted(members, key=int)
        if name == "courtyard" and "3" in candidates:
            candidates.remove("3")
            candidates.insert(0, "3")  # Reuse the previously verified bring-up source first.
        for sequence in candidates[:3]:
            report_path = inspections / f"sequence_{sequence}.json"
            with (inspections / f"sequence_{sequence}.log").open("x") as log:
                result = subprocess.run(command + ["inspect-sequence", sequence, *inputs,
                                                  "--output", str(report_path)],
                                        stdout=log, stderr=subprocess.STDOUT)
            if not report_path.is_file():
                raise RuntimeError(f"sequence {sequence} failed without a report; see inspection log")
            report = json.loads(report_path.read_text())
            if report.get("execution", {}).get("status") != "completed" or not report.get("scope", {}).get("complete"):
                raise RuntimeError(f"incomplete inspection: {report_path}")
            if result.returncode == 0 and report.get("valid") is True:
                if report["sequences"][0]["annotation_count"] < FRAMES_PER_SEQUENCE:
                    raise RuntimeError(f"not enough annotations in {sequence}; no silent smaller export")
                selected[name] = sequence
                print(f"Selected {name}: sequence {sequence}", flush=True)
                break
            if report.get("valid") is not False or not any(f["severity"] == "error" for f in report["findings"]):
                raise RuntimeError(f"unexpected inspection failure: {report_path}")
            print(f"Rejected sequence {sequence}; source findings preserved in {report_path}", flush=True)
        else:
            raise RuntimeError(f"no validated representative among three candidates for {name}")
    (directory / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    sequences = [arg for sequence in selected.values() for arg in ("--sequence", sequence)]
    subprocess.run(command + ["build-manifest", *inputs, *sequences, "--output", str(manifest),
                              "--validation-report", str(directory / "manifest_validation.json")], check=True)
    export(manifest, Path(inputs[1]), directory / "export")
    print(json.dumps(verify_export(directory), sort_keys=True), flush=True)
    print("SCENE DIAGNOSTIC PREPARATION COMPLETE; GPU dependency may start", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    prepare(parser.parse_args().directory)
