"""Canonical Phase 0 command-line workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Phase0Config, load_config
from .dataset import FREDDataset
from .fred_api import HFFredSource
from .inventory import build_inventory, load_inventory, write_inventory
from .manifest import ManifestWriter
from .provenance import atomic_write_json, sha256_file, utc_now
from .schema import AccessMode, Modality, ProjectSplit
from .splits import (
    build_official_split_manifest,
    load_official_split_manifest,
    load_project_split_manifest,
    validate_official_split_manifest,
    validate_project_split,
    write_official_split_manifest,
)
from .validation import inspect_sequence, write_validation_report


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "phase0" / "default.yaml"


def _config(args: argparse.Namespace) -> Phase0Config:
    return load_config(args.config, workspace_root=args.workspace_root)


def _source(config: Phase0Config) -> HFFredSource:
    return HFFredSource(config)


def _inspect_prepared_sequence(
    *,
    source: HFFredSource,
    config: Phase0Config,
    record: object,
    official: object,
    project: object,
):
    sequence_id = record.sequence_id
    source.prepare_sequence(sequence_id, record)
    with source.activate_prepared_window(((sequence_id, record),)):
        with source.open_prepared_sequence(sequence_id, record) as prepared_sequence:
            return inspect_sequence(
                config=config,
                inventory_record=record,
                prepared_sequence=prepared_sequence,
                official_split=official,
                project_split=project,
            )


def command_verify_source(args: argparse.Namespace) -> int:
    config = _config(args)
    result = _source(config).verify_revision()
    result.update(
        {
            "schema_version": "phase0-source-verification-v1",
            "verified_at": utc_now(),
            "fred_repository_url": config.fred_repository.url,
            "fred_repository_revision": config.fred_repository.revision,
        }
    )
    output = Path(args.output) if args.output else config.metadata_root / "source_verification.json"
    atomic_write_json(output, result)
    print(output.resolve())
    return 0


def command_inventory(args: argparse.Namespace) -> int:
    config = _config(args)
    inventory = build_inventory(_source(config))
    output = Path(args.output) if args.output else config.metadata_root / "source_inventory.json"
    write_inventory(inventory, output)
    print(output.resolve())
    return 0


def command_build_official_splits(args: argparse.Namespace) -> int:
    config = _config(args)
    inventory = load_inventory(args.inventory)
    _assert_inventory_config(config, inventory)
    manifest = build_official_split_manifest(config, inventory)
    output = (
        Path(args.output)
        if args.output
        else config.metadata_root / "official_challenging_split.json"
    )
    write_official_split_manifest(manifest, output)
    print(output.resolve())
    return 0


def command_fetch_sequence(args: argparse.Namespace) -> int:
    config = _config(args)
    inventory = load_inventory(args.inventory)
    _assert_inventory_config(config, inventory)
    record = _record(inventory, args.sequence)
    prepared_sequence = _source(config).prepare_sequence(args.sequence, record)
    print(prepared_sequence.sequence_root)
    return 0


def command_inspect_sequence(args: argparse.Namespace) -> int:
    config, inventory, official, project = _load_build_inputs(args)
    record = _record(inventory, args.sequence)
    source = _source(config)
    inspection = _inspect_prepared_sequence(
        source=source,
        config=config,
        record=record,
        official=official,
        project=project,
    )
    output = (
        Path(args.output)
        if args.output
        else config.validation_root / f"sequence_{args.sequence}.json"
    )
    write_validation_report(
        [inspection],
        output,
        context={
            "dataset_revision": config.source.revision,
            "repository_revision": config.fred_repository.revision,
            "project_split_version": project.version,
            "timestamp_verification_status": config.timestamp.verification_status,
        },
        expected_sequence_ids=(args.sequence,),
    )
    if not inspection.is_valid:
        raise RuntimeError(f"sequence {args.sequence} has blocking validation findings; see {output}")
    print(output.resolve())
    return 0


def command_build_manifest(args: argparse.Namespace) -> int:
    config, inventory, official, project = _load_build_inputs(args)
    if args.all_sequences:
        if not args.confirm_full_scan:
            raise ValueError("--all-sequences requires --confirm-full-scan")
        sequence_ids = tuple(record.sequence_id for record in inventory.records)
    else:
        sequence_ids = tuple(dict.fromkeys(args.sequence or ()))
    if not sequence_ids:
        raise ValueError("provide --sequence at least once, or explicitly request --all-sequences")

    destination = Path(args.output) if args.output else config.manifest_root / "canonical.sqlite"
    report_path = config.validation_root / "manifest_build_validation.json"
    inspections = []
    source = _source(config)
    metadata = {
        "manifest_version": config.manifest_version,
        "schema_version": config.schema_version,
        "dataset_id": config.source.dataset_id,
        "dataset_revision": config.source.revision,
        "fred_repository_revision": config.fred_repository.revision,
        "official_split_manifest_hash": official.manifest_hash,
        "project_split_version": project.version,
        "project_split_content_hash": project.content_hash,
        "project_split_approval_reference": project.approval_reference,
        "inventory_hash": inventory.inventory_hash,
        "timestamp_policy": config.timestamp.policy,
        "timestamp_verification_status": config.timestamp.verification_status,
        "complete_inventory": set(sequence_ids) == set(inventory.by_sequence()),
        "annotation_policy": "coordinates.txt_unmodified_v1",
        "config_sha256": sha256_file(config.config_path),
    }
    metadata["validation_status"] = (
        "passed_complete_inventory"
        if metadata["complete_inventory"]
        else "passed_selected_sequences_non_freeze"
    )
    execution_error = None
    try:
        with ManifestWriter(destination, metadata) as writer:
            for sequence_id in sorted(sequence_ids, key=int):
                record = _record(inventory, sequence_id)
                inspection = _inspect_prepared_sequence(
                    source=source,
                    config=config,
                    record=record,
                    official=official,
                    project=project,
                )
                inspections.append(inspection)
                if not inspection.is_valid:
                    raise RuntimeError(
                        f"sequence {sequence_id} has blocking findings; manifest was not published"
                    )
                for sample in inspection.samples:
                    writer.add_sample(sample)
    except BaseException as error:
        execution_error = {
            "type": type(error).__name__,
            "message": str(error)[:1000],
        }
        raise
    finally:
        report_context = dict(metadata)
        if execution_error is not None:
            report_context["validation_status"] = "failed_before_publish"
        write_validation_report(
            inspections,
            report_path,
            context=report_context,
            expected_sequence_ids=sequence_ids,
            execution_error=execution_error,
        )
    print(destination.resolve())
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    config, inventory, official, project = _load_build_inputs(args)
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.sequence not in project.train:
        raise ValueError("smoke sequence must belong to approved project training membership")
    source = _source(config)
    record = _record(inventory, args.sequence)
    source.prepare_sequence(args.sequence, record)
    with source.activate_prepared_window(((args.sequence, record),)):
        with source.open_prepared_sequence(args.sequence, record) as prepared_sequence:
            inspection = inspect_sequence(
                config=config,
                inventory_record=record,
                prepared_sequence=prepared_sequence,
                official_split=official,
                project_split=project,
            )
        if not inspection.is_valid:
            raise RuntimeError("smoke sequence has blocking validation findings")
        output = config.manifest_root / "smoke.sqlite"
        metadata = {
            "manifest_version": f"{config.manifest_version}-smoke",
            "schema_version": config.schema_version,
            "dataset_id": config.source.dataset_id,
            "dataset_revision": config.source.revision,
            "fred_repository_revision": config.fred_repository.revision,
            "annotation_policy": "coordinates.txt_unmodified_v1",
            "config_sha256": sha256_file(config.config_path),
            "validation_status": "passed_smoke_scope_non_reportable",
            "non_reportable_toy_artifact": True,
        }
        with ManifestWriter(output, metadata) as writer:
            for sample in inspection.samples[: args.sample_count]:
                writer.add_sample(sample)
        dataset = FREDDataset(
            manifest_path=output,
            inventory=inventory,
            source=source,
            project_split=ProjectSplit.TRAIN,
            modality=Modality.RGB_EVENT,
            access_mode=AccessMode.TRAIN,
        )
        loaded = dataset[0]
    report = {
        "schema_version": "phase0-smoke-report-v1",
        "created_at": utc_now(),
        "sequence_id": args.sequence,
        "sample_id": loaded.sample_id,
        "rgb_size": loaded.rgb.size if loaded.rgb else None,
        "event_size": loaded.event.size if loaded.event else None,
        "annotation_count": len(loaded.annotations or ()),
        "passed": True,
        "reportable_result": False,
    }
    report_path = config.validation_root / "smoke_report.json"
    atomic_write_json(report_path, report)
    print(report_path.resolve())
    return 0


def _assert_inventory_config(config: Phase0Config, inventory: object) -> None:
    if inventory.dataset_id != config.source.dataset_id:
        raise ValueError("inventory dataset identity differs from configuration")
    if inventory.dataset_revision != config.source.revision:
        raise ValueError("inventory dataset revision differs from configuration")


def _record(inventory: object, sequence_id: str):
    try:
        return inventory.by_sequence()[sequence_id]
    except KeyError as error:
        raise ValueError(f"sequence {sequence_id} is absent from the pinned inventory") from error


def _load_build_inputs(args: argparse.Namespace):
    config = _config(args)
    inventory = load_inventory(args.inventory)
    _assert_inventory_config(config, inventory)
    official = load_official_split_manifest(args.official_split)
    project = load_project_split_manifest(args.project_split)
    if official.repository_revision != config.fred_repository.revision:
        raise ValueError("official split repository revision differs from configuration")
    if official.train_source_sha256 != config.fred_repository.challenging_train_sha256:
        raise ValueError("official training split checksum differs from configuration")
    if official.test_source_sha256 != config.fred_repository.challenging_test_sha256:
        raise ValueError("official test split checksum differs from configuration")
    validate_official_split_manifest(official, inventory)
    validate_project_split(project, official)
    return config, inventory, official, project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FRED Phase 0 data infrastructure")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-source")
    verify.add_argument("--output", type=Path)
    verify.set_defaults(handler=command_verify_source)

    inventory = commands.add_parser("inventory")
    inventory.add_argument("--output", type=Path)
    inventory.set_defaults(handler=command_inventory)

    official = commands.add_parser("build-official-splits")
    official.add_argument("--inventory", type=Path, required=True)
    official.add_argument("--output", type=Path)
    official.set_defaults(handler=command_build_official_splits)

    fetch = commands.add_parser("fetch-sequence")
    fetch.add_argument("sequence")
    fetch.add_argument("--inventory", type=Path, required=True)
    fetch.set_defaults(handler=command_fetch_sequence)

    inspect = commands.add_parser("inspect-sequence")
    _add_build_inputs(inspect)
    inspect.add_argument("sequence")
    inspect.add_argument("--output", type=Path)
    inspect.set_defaults(handler=command_inspect_sequence)

    manifest = commands.add_parser("build-manifest")
    _add_build_inputs(manifest)
    manifest.add_argument("--sequence", action="append")
    manifest.add_argument("--all-sequences", action="store_true")
    manifest.add_argument("--confirm-full-scan", action="store_true")
    manifest.add_argument("--output", type=Path)
    manifest.set_defaults(handler=command_build_manifest)

    smoke = commands.add_parser("smoke")
    _add_build_inputs(smoke)
    smoke.add_argument("sequence")
    smoke.add_argument("--sample-count", type=int, default=2)
    smoke.set_defaults(handler=command_smoke)
    return parser


def _add_build_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--project-split", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
