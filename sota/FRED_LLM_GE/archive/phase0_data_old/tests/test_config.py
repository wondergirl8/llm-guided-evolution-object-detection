from pathlib import Path

import pytest

from sota.FRED_LLM_GE.phase0_data.config import load_config


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "phase0" / "default.yaml"


def test_default_config_has_full_pins_and_bounded_cache(tmp_path: Path):
    config = load_config(CONFIG, workspace_root=tmp_path)
    assert len(config.source.revision) == 40
    assert len(config.fred_repository.revision) == 40
    assert config.cache.max_bytes == 50 * 1024**3
    assert config.cache.root.is_relative_to(tmp_path)


def test_string_cache_verification_flag_is_rejected(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "verify_content_sha256: false", 'verify_content_sha256: "false"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="YAML boolean"):
        load_config(path, workspace_root=tmp_path)


def test_abbreviated_source_revision_is_rejected(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    source = CONFIG.read_text(encoding="utf-8")
    path.write_text(
        source.replace(
            "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd",
            "980a8fa0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="full 40-character"):
        load_config(path, workspace_root=tmp_path)
