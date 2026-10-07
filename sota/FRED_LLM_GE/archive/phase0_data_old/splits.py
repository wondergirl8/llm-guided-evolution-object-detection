"""Protected official split parsing and approved project-split validation."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi

from .config import Phase0Config
from .inventory import SourceInventory
from .provenance import atomic_write_bytes, atomic_write_json, sha256_bytes, stable_hash, utc_now


@dataclass(frozen=True, slots=True)
class OfficialSplitManifest:
    repository_revision: str
    challenging_train: tuple[str, ...]
    challenging_test: tuple[str, ...]
    train_source_sha256: str
    test_source_sha256: str
    manifest_hash: str
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "fred-official-challenging-split-v1",
            "repository_revision": self.repository_revision,
            "challenging_train": list(self.challenging_train),
            "challenging_test": list(self.challenging_test),
            "train_source_sha256": self.train_source_sha256,
            "test_source_sha256": self.test_source_sha256,
            "manifest_hash": self.manifest_hash,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ProjectSplitManifest:
    version: str
    grouping_unit: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    held_out_test: tuple[str, ...]
    approval_reference: str

    @property
    def content_hash(self) -> str:
        return stable_hash(
            {
                "version": self.version,
                "grouping_unit": self.grouping_unit,
                "train": self.train,
                "validation": self.validation,
                "held_out_test": self.held_out_test,
                "approval_reference": self.approval_reference,
            }
        )


def parse_split_text(content: str, *, source: str) -> tuple[str, ...]:
    members: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(content.splitlines(), start=1):
        value = raw.strip().strip("/").strip()
        if not value:
            continue
        if not value.isdecimal():
            raise ValueError(f"{source}:{line_number}: invalid sequence ID {value!r}")
        if value in seen:
            raise ValueError(f"{source}:{line_number}: duplicate sequence ID {value}")
        seen.add(value)
        members.append(value)
    if not members:
        raise ValueError(f"{source}: split is empty")
    return tuple(members)


def _fetch_verified(
    url: str,
    expected_sha256: str,
    destination: Path,
    *,
    retries: int,
    backoff_seconds: float,
) -> str:
    if destination.is_file():
        content = destination.read_bytes()
        if sha256_bytes(content) == expected_sha256:
            return content.decode("utf-8")
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fred-phase0/1"})
            tls_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(request, timeout=30, context=tls_context) as response:
                content = response.read()
            digest = sha256_bytes(content)
            if digest != expected_sha256:
                raise ValueError(
                    f"protected split checksum mismatch: expected {expected_sha256}, got {digest}"
                )
            atomic_write_bytes(destination, content)
            return content.decode("utf-8")
        except (OSError, UnicodeError, ValueError, urllib.error.URLError) as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise RuntimeError(f"failed to retrieve protected split: {'; '.join(errors)}")


def build_official_split_manifest(
    config: Phase0Config, inventory: SourceInventory
) -> OfficialSplitManifest:
    if inventory.dataset_id != config.source.dataset_id:
        raise ValueError("inventory dataset identity differs from configuration")
    if inventory.dataset_revision != config.source.revision:
        raise ValueError("inventory dataset revision differs from configuration")
    split_cache = config.metadata_root / "official_splits" / config.fred_repository.revision
    train_text = _fetch_verified(
        config.fred_repository.challenging_train_url,
        config.fred_repository.challenging_train_sha256,
        split_cache / "challenging_train_split.txt",
        retries=config.source.max_retries,
        backoff_seconds=config.source.retry_backoff_seconds,
    )
    test_text = _fetch_verified(
        config.fred_repository.challenging_test_url,
        config.fred_repository.challenging_test_sha256,
        split_cache / "challenging_test_split.txt",
        retries=config.source.max_retries,
        backoff_seconds=config.source.retry_backoff_seconds,
    )
    train = parse_split_text(train_text, source="challenging_train_split.txt")
    test = parse_split_text(test_text, source="challenging_test_split.txt")
    _validate_membership(train, test, set(inventory.by_sequence()))
    payload = {"train": train, "test": test, "repository_revision": config.fred_repository.revision}
    return OfficialSplitManifest(
        repository_revision=config.fred_repository.revision,
        challenging_train=train,
        challenging_test=test,
        train_source_sha256=config.fred_repository.challenging_train_sha256,
        test_source_sha256=config.fred_repository.challenging_test_sha256,
        manifest_hash=stable_hash(payload),
        created_at=utc_now(),
    )


def _validate_membership(
    train: tuple[str, ...], test: tuple[str, ...], inventory_ids: set[str]
) -> None:
    train_set, test_set = set(train), set(test)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"official challenging split overlaps: {sorted(overlap)}")
    listed = train_set | test_set
    missing = inventory_ids - listed
    unknown = listed - inventory_ids
    if missing or unknown:
        raise ValueError(
            f"official split/inventory mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_project_split(
    project: ProjectSplitManifest, official: OfficialSplitManifest
) -> None:
    if not project.version:
        raise ValueError("project split requires a non-empty version")
    _validate_unique_decimal_members(project.train, "project train")
    _validate_unique_decimal_members(project.validation, "project validation")
    _validate_unique_decimal_members(project.held_out_test, "project held-out test")
    train, validation, test = set(project.train), set(project.validation), set(project.held_out_test)
    if not project.approval_reference:
        raise ValueError("project split requires a non-empty DG-P0-02 approval reference")
    if project.grouping_unit not in {"sequence", "recording_group"}:
        raise ValueError("project split grouping unit must be sequence or recording_group")
    if not train or not validation or not test:
        raise ValueError("project train, validation, and held-out test must all be non-empty")
    if train & validation or train & test or validation & test:
        raise ValueError("project train/validation/test sequence membership overlaps")
    official_train = set(official.challenging_train)
    official_test = set(official.challenging_test)
    if train | validation != official_train:
        raise ValueError("project train and validation must partition challenging-train exactly")
    if test != official_test:
        raise ValueError("project held-out test must equal official challenging-test exactly")


def validate_official_split_manifest(
    manifest: OfficialSplitManifest, inventory: SourceInventory
) -> None:
    _validate_unique_decimal_members(manifest.challenging_train, "official challenging-train")
    _validate_unique_decimal_members(manifest.challenging_test, "official challenging-test")
    _validate_membership(
        manifest.challenging_train,
        manifest.challenging_test,
        set(inventory.by_sequence()),
    )


def _validate_unique_decimal_members(members: tuple[str, ...], name: str) -> None:
    if len(members) != len(set(members)):
        raise ValueError(f"{name} contains duplicate sequence IDs")
    invalid = [member for member in members if not member.isdecimal()]
    if invalid:
        raise ValueError(f"{name} contains invalid sequence IDs: {invalid}")


def write_official_split_manifest(manifest: OfficialSplitManifest, path: str | Path) -> None:
    atomic_write_json(path, manifest.to_dict())


def load_official_split_manifest(path: str | Path) -> OfficialSplitManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "fred-official-challenging-split-v1":
        raise ValueError(f"unsupported official split schema {raw.get('schema_version')!r}")
    manifest = OfficialSplitManifest(
        repository_revision=raw["repository_revision"],
        challenging_train=tuple(raw["challenging_train"]),
        challenging_test=tuple(raw["challenging_test"]),
        train_source_sha256=raw["train_source_sha256"],
        test_source_sha256=raw["test_source_sha256"],
        manifest_hash=raw["manifest_hash"],
        created_at=raw["created_at"],
    )
    expected = stable_hash(
        {
            "train": manifest.challenging_train,
            "test": manifest.challenging_test,
            "repository_revision": manifest.repository_revision,
        }
    )
    if expected != manifest.manifest_hash:
        raise ValueError("official split manifest hash mismatch")
    return manifest


def load_project_split_manifest(path: str | Path) -> ProjectSplitManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectSplitManifest(
        version=raw["version"],
        grouping_unit=raw["grouping_unit"],
        train=tuple(raw["train"]),
        validation=tuple(raw["validation"]),
        held_out_test=tuple(raw["held_out_test"]),
        approval_reference=raw["approval_reference"],
    )
