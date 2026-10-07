"""Validated Phase 0 configuration loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml


_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SourceConfig:
    provider: str
    dataset_id: str
    revision: str
    repo_type: str
    token_environment_variable: str
    max_retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class RepositoryConfig:
    url: str
    revision: str
    challenging_train_url: str
    challenging_train_sha256: str
    challenging_test_url: str
    challenging_test_sha256: str


@dataclass(frozen=True, slots=True)
class CacheConfig:
    root: Path
    max_bytes: int
    lock_timeout_seconds: float
    verify_content_sha256: bool


@dataclass(frozen=True, slots=True)
class TimestampConfig:
    policy: str
    frame_period_seconds: str
    frame_index_offset: int
    tolerance_seconds: str
    verification_status: str


@dataclass(frozen=True, slots=True)
class Phase0Config:
    config_path: Path
    schema_version: str
    manifest_version: str
    source: SourceConfig
    fred_repository: RepositoryConfig
    runtime_root: Path
    metadata_root: Path
    manifest_root: Path
    validation_root: Path
    cache: CacheConfig
    timestamp: TimestampConfig


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key {context}.{key}")
    return mapping[key]


def _resolve_runtime_root(raw: str, *, workspace_root: Path) -> Path:
    override = os.getenv("FRED_PHASE0_ROOT")
    value = Path(override if override else raw).expanduser()
    return (value if value.is_absolute() else workspace_root / value).resolve()


def load_config(path: str | Path, *, workspace_root: str | Path | None = None) -> Phase0Config:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Phase 0 configuration must be a mapping")
    workspace = Path(workspace_root or Path.cwd()).expanduser().resolve()

    source_raw = _required(raw, "source", "phase0")
    repository_raw = _required(raw, "fred_repository", "phase0")
    cache_raw = _required(raw, "cache", "phase0")
    timestamp_raw = _required(raw, "timestamp", "phase0")
    if not all(isinstance(value, dict) for value in (source_raw, repository_raw, cache_raw, timestamp_raw)):
        raise ValueError("source, fred_repository, cache, and timestamp must be mappings")

    runtime_root = _resolve_runtime_root(
        str(_required(raw, "runtime_root", "phase0")), workspace_root=workspace
    )
    max_bytes = int(_required(cache_raw, "max_bytes", "cache"))
    if max_bytes <= 0:
        raise ValueError("cache.max_bytes must be positive")

    source = SourceConfig(
        provider=str(_required(source_raw, "provider", "source")),
        dataset_id=str(_required(source_raw, "dataset_id", "source")),
        revision=str(_required(source_raw, "revision", "source")),
        repo_type=str(_required(source_raw, "repo_type", "source")),
        token_environment_variable=str(
            _required(source_raw, "token_environment_variable", "source")
        ),
        max_retries=int(_required(source_raw, "max_retries", "source")),
        retry_backoff_seconds=float(
            _required(source_raw, "retry_backoff_seconds", "source")
        ),
    )
    if source.provider != "huggingface_hub" or source.repo_type != "dataset":
        raise ValueError("Phase 0 v1 currently supports only a Hugging Face dataset source")
    if not _COMMIT_PATTERN.fullmatch(source.revision):
        raise ValueError("source.revision must be a lowercase full 40-character commit hash")
    if source.max_retries <= 0 or source.retry_backoff_seconds < 0:
        raise ValueError("source retry count must be positive and backoff non-negative")

    repository = RepositoryConfig(
        url=str(_required(repository_raw, "url", "fred_repository")),
        revision=str(_required(repository_raw, "revision", "fred_repository")),
        challenging_train_url=str(
            _required(repository_raw, "challenging_train_url", "fred_repository")
        ),
        challenging_train_sha256=str(
            _required(repository_raw, "challenging_train_sha256", "fred_repository")
        ),
        challenging_test_url=str(
            _required(repository_raw, "challenging_test_url", "fred_repository")
        ),
        challenging_test_sha256=str(
            _required(repository_raw, "challenging_test_sha256", "fred_repository")
        ),
    )
    if not _COMMIT_PATTERN.fullmatch(repository.revision):
        raise ValueError("fred_repository.revision must be a lowercase full commit hash")
    for name, value in (
        ("challenging_train_sha256", repository.challenging_train_sha256),
        ("challenging_test_sha256", repository.challenging_test_sha256),
    ):
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"fred_repository.{name} must be a lowercase SHA-256 digest")
    for name, value in (
        ("url", repository.url),
        ("challenging_train_url", repository.challenging_train_url),
        ("challenging_test_url", repository.challenging_test_url),
    ):
        if not value.startswith("https://"):
            raise ValueError(f"fred_repository.{name} must use HTTPS")

    verify_sha256 = _required(cache_raw, "verify_content_sha256", "cache")
    if not isinstance(verify_sha256, bool):
        raise ValueError("cache.verify_content_sha256 must be a YAML boolean")
    lock_timeout = float(_required(cache_raw, "lock_timeout_seconds", "cache"))
    if lock_timeout <= 0:
        raise ValueError("cache.lock_timeout_seconds must be positive")
    cache_relative_root = Path(str(_required(cache_raw, "relative_root", "cache")))
    if (
        cache_relative_root.is_absolute()
        or ".." in cache_relative_root.parts
        or cache_relative_root == Path(".")
    ):
        raise ValueError("cache.relative_root must be a non-empty contained relative path")
    cache = CacheConfig(
        root=(runtime_root / cache_relative_root).resolve(),
        max_bytes=max_bytes,
        lock_timeout_seconds=lock_timeout,
        verify_content_sha256=verify_sha256,
    )
    timestamp = TimestampConfig(
        policy=str(_required(timestamp_raw, "policy", "timestamp")),
        frame_period_seconds=str(
            _required(timestamp_raw, "frame_period_seconds", "timestamp")
        ),
        frame_index_offset=int(
            _required(timestamp_raw, "frame_index_offset", "timestamp")
        ),
        tolerance_seconds=str(_required(timestamp_raw, "tolerance_seconds", "timestamp")),
        verification_status=str(
            _required(timestamp_raw, "verification_status", "timestamp")
        ),
    )
    try:
        frame_period = Decimal(timestamp.frame_period_seconds)
        tolerance = Decimal(timestamp.tolerance_seconds)
    except InvalidOperation as error:
        raise ValueError("timestamp period and tolerance must be decimal values") from error
    if not frame_period.is_finite() or frame_period <= 0:
        raise ValueError("timestamp.frame_period_seconds must be finite and positive")
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("timestamp.tolerance_seconds must be finite and non-negative")
    if timestamp.frame_index_offset < 0:
        raise ValueError("timestamp.frame_index_offset must be non-negative")
    return Phase0Config(
        config_path=config_path,
        schema_version=str(_required(raw, "schema_version", "phase0")),
        manifest_version=str(_required(raw, "manifest_version", "phase0")),
        source=source,
        fred_repository=repository,
        runtime_root=runtime_root,
        metadata_root=runtime_root / "metadata",
        manifest_root=runtime_root / "manifests",
        validation_root=runtime_root / "validation",
        cache=cache,
        timestamp=timestamp,
    )
