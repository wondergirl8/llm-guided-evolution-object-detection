import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sota.FRED_LLM_GE.phase0_data import cli
from sota.FRED_LLM_GE.phase0_data.schema import RemoteObject
from sota.FRED_LLM_GE.phase0_data.splits import ProjectSplitManifest


class FailingSource:
    def prepare_sequence(self, sequence_id, record):
        raise RuntimeError("representative fetch failure")


def test_failed_manifest_fetch_writes_failed_incomplete_report(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: config\n", encoding="utf-8")
    record = SimpleNamespace(
        sequence_id="0",
        archive=RemoteObject("hf://archive", "train/0.zip"),
    )
    inventory = SimpleNamespace(
        records=(record,),
        inventory_hash="inventory-hash",
        by_sequence=lambda: {"0": record},
    )
    config = SimpleNamespace(
        config_path=config_path,
        manifest_root=tmp_path / "manifests",
        validation_root=tmp_path / "validation",
        manifest_version="manifest-v1",
        schema_version="schema-v1",
        source=SimpleNamespace(dataset_id="fred", revision="a" * 40),
        fred_repository=SimpleNamespace(revision="b" * 40),
        timestamp=SimpleNamespace(policy="policy-v1", verification_status="pending"),
    )
    official = SimpleNamespace(manifest_hash="official-hash")
    project = ProjectSplitManifest(
        "project-v1", "sequence", ("0",), ("1",), ("2",), "DG-P0-02"
    )
    monkeypatch.setattr(
        cli,
        "_load_build_inputs",
        lambda args: (config, inventory, official, project),
    )
    monkeypatch.setattr(cli, "_source", lambda _: FailingSource())
    args = SimpleNamespace(
        all_sequences=False,
        confirm_full_scan=False,
        sequence=["0"],
        output=tmp_path / "manifest.sqlite",
    )

    with pytest.raises(RuntimeError, match="representative fetch failure"):
        cli.command_build_manifest(args)

    report_path = tmp_path / "validation" / "manifest_build_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["status"] == "failed"
    assert report["execution"]["error"]["type"] == "RuntimeError"
    assert report["scope"]["missing_sequence_ids"] == ["0"]
    assert not args.output.exists()
