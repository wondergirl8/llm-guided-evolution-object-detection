import pytest

from sota.FRED_LLM_GE.phase1_detection.result_contract import (
    DetectionResult,
    validate_gene_id,
    write_results,
)


def test_writes_exact_orchestrator_csv_row(tmp_path):
    result = DetectionResult(map50=0.8, map50_95=0.6, params=123456)
    output = write_results(tmp_path, "gene_1", result)
    assert output == tmp_path / "results" / "gene_1_results.csv"
    # No header is written; run_improved.py reads these values by position.
    assert output.read_text() == "0.800000,0.600000,123456,\n"


def test_failure_row_has_invalid_fitness_and_reason(tmp_path):
    import csv
    import math

    failed = DetectionResult(-math.inf, -math.inf, math.inf)
    output = write_results(tmp_path, "bad_gene", failed, reason="build_failed: bad, YAML")
    with output.open(newline="") as file:
        assert next(csv.reader(file)) == ["-inf", "-inf", "inf", "build_failed: bad, YAML"]
    with pytest.raises(ValueError, match="requires a reason"):
        write_results(tmp_path, "bad_gene", failed)


@pytest.mark.parametrize("gene_id", ["", "../escape", "bad/name", " space"])
def test_rejects_unsafe_gene_ids(gene_id):
    with pytest.raises(ValueError):
        validate_gene_id(gene_id)


@pytest.mark.parametrize(
    "values",
    [(-0.1, 0.5, 1), (1.1, 0.5, 1), (0.5, -0.1, 1), (0.5, 0.5, -1)],
)
def test_rejects_invalid_result_values(values):
    with pytest.raises(ValueError):
        DetectionResult(*values)
