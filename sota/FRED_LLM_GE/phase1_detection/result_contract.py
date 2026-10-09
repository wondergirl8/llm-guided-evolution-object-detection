"""Define the evaluation result and write the orchestrator CSV file."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True)
class DetectionResult:
    """Final fitness values for one evaluated model."""

    map50: float
    map50_95: float
    params: int


    # validating the fields of the dataclass
    def __post_init__(self):
        if (
            isinstance(self.map50, bool)
            or not isinstance(self.map50, Real)
            or not 0 <= self.map50 <= 1
        ):
            raise ValueError("map50 must be a number between 0 and 1")

        if (
            isinstance(self.map50_95, bool)
            or not isinstance(self.map50_95, Real)
            or not 0 <= self.map50_95 <= 1
        ):
            raise ValueError("map50_95 must be a number between 0 and 1")

        if (
            isinstance(self.params, bool)
            or not isinstance(self.params, Integral)
            or self.params < 0
        ):
            raise ValueError("params must be a nonnegative integer")


_GENE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def validate_gene_id(gene_id):
    """Reject unsafe or malformed identifiers before using one in a filename."""

    if not isinstance(gene_id, str):
        raise ValueError("gene_id must be a string")

    if not gene_id:
        raise ValueError("gene_id must not be empty")

    if len(gene_id) > 128:
        raise ValueError("gene_id must contain at most 128 characters")

    if not _GENE_ID_PATTERN.fullmatch(gene_id):
        raise ValueError(
            "gene_id may contain only letters, numbers, periods, hyphens, "
            "and underscores, and must start with a letter or number"
        )

    return gene_id


def write_results(sota_root, gene_id, result):
    """Write the CSV row consumed by run_improved.py."""

    validate_gene_id(gene_id)

    if not isinstance(result, DetectionResult):
        raise TypeError("result must be a DetectionResult")

    directory = Path(sota_root) / "results"
    directory.mkdir(parents=True, exist_ok=True)

    output = directory / f"{gene_id}_results.csv"
    temporary_name = None

    try:
        with NamedTemporaryFile(
            mode="w",
            newline="",
            dir=directory,
            delete=False,
        ) as handle:
            temporary_name = handle.name

            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    f"{result.map50:.6f}",
                    f"{result.map50_95:.6f}",
                    str(result.params),
                ]
            )

        os.replace(temporary_name, output)

    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)

    return output