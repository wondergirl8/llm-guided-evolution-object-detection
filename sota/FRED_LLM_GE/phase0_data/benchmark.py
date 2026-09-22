"""Reproducible cold/warm Phase 0 loader benchmark records."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, utc_now


def benchmark_samples(
    operation: Callable[[int], Any],
    *,
    sample_count: int,
    context: dict[str, Any],
    destination: str | Path,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    started = time.perf_counter()
    for index in range(sample_count):
        operation(index)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "phase0-performance-report-v1",
        "created_at": utc_now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "context": context,
        "sample_count": sample_count,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / elapsed if elapsed else None,
    }
    atomic_write_json(destination, report)
    return report
