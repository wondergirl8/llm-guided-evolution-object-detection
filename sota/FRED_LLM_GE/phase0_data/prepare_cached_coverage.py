"""Validate cached recordings fully and create a fresh bounded derived export."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from sota.FRED_LLM_GE.archive.phase0_data_old.annotation_policy import POLICY_VERSION
from sota.FRED_LLM_GE.archive.phase0_data_old.config import load_config
from sota.FRED_LLM_GE.archive.phase0_data_old.inventory import load_inventory
from sota.FRED_LLM_GE.data.cached_source import CachedOnlyFredSource
from sota.FRED_LLM_GE.data.scene_coverage_export import COHORT, selected_sequences, export, verify_export
from .check_scene_split import DEFAULT_PROJECT, main as check_scene_split

INVENTORY = Path("data/fred_phase0/metadata/source_inventory.json")
OFFICIAL = Path("data/fred_phase0/metadata/official_challenging_split.json")
CONFIG = Path("sota/FRED_LLM_GE/configs/phase0/default.yaml")


def cached_readiness():
    """Cheap submission gate; full cache/content checks happen on the CPU job."""
    config = load_config(CONFIG)
    inventory = load_inventory(INVENTORY)
    if (inventory.dataset_id, inventory.dataset_revision) != (config.source.dataset_id, config.source.revision):
        raise ValueError("inventory differs from pinned source")
    selected = selected_sequences()
    entries = {}
    for path in config.cache.root.glob("entries/*/entry.json"):
        record = json.loads(path.read_text())
        identity = record.get("identity", {})
        sequence = record.get("entry_metadata", {}).get("sequence_id")
        if (identity.get("provider") == "phase0_prepared_sequence" and sequence in selected
                and identity.get("dataset_id") == inventory.dataset_id
                and identity.get("revision") == inventory.dataset_revision):
            if sequence in entries:
                raise ValueError(f"ambiguous prepared cache entries for {sequence}")
            entries[sequence] = str(path.resolve())
    missing = set(selected) - set(entries)
    if missing:
        raise ValueError(f"missing prepared cache entries: {sorted(missing, key=int)}; no downloads allowed")
    return {"selected_sequences": selected, "cache_entries": entries}


def prepare(directory):
    check_scene_split([])
    readiness = cached_readiness()
    directory = Path(directory)
    if any((directory / name).exists() for name in ("manifest.sqlite", "export", "preparation.json")):
        raise FileExistsError("refusing to reuse coverage preparation outputs")
    config = load_config(CONFIG)
    inventory = load_inventory(INVENTORY)
    source = CachedOnlyFredSource(config)
    selected = readiness["selected_sequences"]
    # Validate all handles before building; never create or evict a cache entry.
    with source.activate_prepared_window((s, inventory.by_sequence()[s]) for s in selected):
        pass
    (directory / "preparation.json").write_text(json.dumps({**readiness, "cohort": COHORT,
                                                          "cached_only": True}, indent=2) + "\n")
    command = [sys.executable, "-m", "sota.FRED_LLM_GE.archive.phase0_data_old", "build-manifest",
               "--inventory", str(INVENTORY), "--official-split", str(OFFICIAL),
               "--project-split", str(DEFAULT_PROJECT), "--annotation-policy", POLICY_VERSION,
               "--cached-only", "--output", str(directory / "manifest.sqlite"),
               "--validation-report", str(directory / "manifest_validation.json")]
    command.extend(arg for s in sorted(selected, key=int) for arg in ("--sequence", s))
    subprocess.run(command, check=True)
    export(directory / "manifest.sqlite", INVENTORY, directory / "export")
    print(json.dumps(verify_export(directory), indent=2), flush=True)
    print("CACHED COVERAGE PREPARATION COMPLETE; no downloads; GPU dependency may start", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    prepare(parser.parse_args().directory)
