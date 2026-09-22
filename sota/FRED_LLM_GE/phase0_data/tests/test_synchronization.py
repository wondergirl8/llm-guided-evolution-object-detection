from pathlib import Path

from PIL import Image

from sota.FRED_LLM_GE.phase0_data.config import TimestampConfig
from sota.FRED_LLM_GE.phase0_data.synchronization import (
    pair_sequence_frames,
    timestamp_for_frame,
)


POLICY = TimestampConfig(
    "fred_upstream_30hz_v1", "0.033333", 1, "0.000001", "pending"
)


def test_timestamp_policy_reproduces_pinned_upstream_formula():
    assert timestamp_for_frame(0, POLICY) == "0.033333"
    assert timestamp_for_frame(39, POLICY) == "1.33332"


def test_pairing_uses_natural_order(sequence_root: Path):
    result = pair_sequence_frames(sequence_root, sequence_id="0", timestamp_policy=POLICY)
    assert not result.findings
    assert [item.rgb_relative_path for item in result.pairs] == [
        "RGB/Video_0_00_00_00.000000.jpg",
        "RGB/Video_0_00_00_00.033333.jpg",
    ]


def test_pairing_count_mismatch_is_blocking(sequence_root: Path):
    (sequence_root / "Event" / "Frames" / "Video_0_frame_66666.png").unlink()
    result = pair_sequence_frames(sequence_root, sequence_id="0", timestamp_policy=POLICY)
    assert result.pairs == ()
    assert (result.rgb_count, result.event_count) == (2, 1)
    assert result.findings[0].code == "pairing.count_mismatch"


def test_equal_modality_counts_with_shared_interior_gap_are_blocking(
    sequence_root: Path,
):
    Image.new("RGB", (32, 24), "black").save(
        sequence_root / "RGB" / "Video_0_00_00_00.066666.jpg"
    )
    Image.new("RGB", (32, 24), "white").save(
        sequence_root / "Event" / "Frames" / "Video_0_frame_99999.png"
    )
    (sequence_root / "RGB" / "Video_0_00_00_00.033333.jpg").unlink()
    (sequence_root / "Event" / "Frames" / "Video_0_frame_66666.png").unlink()

    result = pair_sequence_frames(sequence_root, sequence_id="0", timestamp_policy=POLICY)

    assert result.pairs == ()
    assert (result.rgb_count, result.event_count) == (2, 2)
    assert any(
        finding.code == "pairing.event_counter_gap_or_offset"
        for finding in result.findings
    )


def test_unrecognized_event_filename_is_blocking(sequence_root: Path):
    source = sequence_root / "Event" / "Frames" / "Video_0_frame_66666.png"
    source.rename(source.with_name("unverified_66666.png"))

    result = pair_sequence_frames(sequence_root, sequence_id="0", timestamp_policy=POLICY)

    assert result.pairs == ()
    assert any(
        finding.code == "pairing.event_filename_unrecognized"
        for finding in result.findings
    )


def test_duplicate_event_counter_is_blocking(sequence_root: Path):
    Image.new("RGB", (32, 24), "black").save(
        sequence_root / "RGB" / "Video_0_00_00_00.066666.jpg"
    )
    Image.new("RGB", (32, 24), "white").save(
        sequence_root / "Event" / "Frames" / "Video_0_frame_033333.png"
    )

    result = pair_sequence_frames(sequence_root, sequence_id="0", timestamp_policy=POLICY)

    assert result.pairs == ()
    assert any(
        finding.code == "pairing.event_duplicate_counter"
        for finding in result.findings
    )


def test_unsupported_file_in_frame_directory_is_blocking(sequence_root: Path):
    (sequence_root / "Event" / "Frames" / "notes.txt").write_text(
        "not a released frame", encoding="utf-8"
    )

    result = pair_sequence_frames(sequence_root, sequence_id="0", timestamp_policy=POLICY)

    assert result.pairs == ()
    assert any(
        finding.code == "pairing.unsupported_frame_files"
        for finding in result.findings
    )
