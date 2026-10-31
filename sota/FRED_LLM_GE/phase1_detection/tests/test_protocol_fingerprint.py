from types import SimpleNamespace

import pytest
import yaml

from sota.FRED_LLM_GE.phase1_detection import protocol_fingerprint


def test_fingerprint_changes_with_protocol_and_search_manifest(tmp_path, monkeypatch):
    fred = tmp_path / "sota" / "FRED_LLM_GE"
    protocol_path = fred / "configs" / "phase1" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol = {"families": {"yolo11": {"data": "challenging_search"}},
                "challenging": {"data_root": "cache",
                                "search": {"train_frame_step": 50, "val_frame_step": 10}}}
    protocol_path.write_text(yaml.safe_dump(protocol))
    seed = fred / "seeds" / "seed.yaml"
    seed.parent.mkdir()
    seed.write_text("backbone: []\nhead: []\n")
    cache = fred / "cache"
    for relative in ("splits/challenging_train_split.txt",
                     "search_train_step50/train.txt", "search_val_step10/train.txt"):
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("one\n")
    monkeypatch.setattr(protocol_fingerprint.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="commit-1\n"))
    monkeypatch.setattr(protocol_fingerprint.importlib.metadata, "version", lambda _: "8.4.175")

    first = protocol_fingerprint.compute_fingerprint(tmp_path, "yolo11", "run_1", seed)
    assert first["ultralytics_version"] == "8.4.175"
    assert "challenging_test" not in first["files"]
    assert first == protocol_fingerprint.compute_fingerprint(tmp_path, "yolo11", "run_1", seed)

    (cache / "search_val_step10" / "train.txt").write_text("two\n")
    assert first["sha256"] != protocol_fingerprint.compute_fingerprint(
        tmp_path, "yolo11", "run_1", seed)["sha256"]
    assert first["sha256"] != protocol_fingerprint.compute_fingerprint(
        tmp_path, "yolo11", "run_2", seed)["sha256"]


def test_resume_rejects_missing_or_changed_fingerprint():
    expected = {"sha256": "first"}
    protocol_fingerprint.verify_fingerprint({"protocol_fingerprint": expected}, "checkpoint", expected)
    with pytest.raises(RuntimeError, match="missing or mismatched"):
        protocol_fingerprint.verify_fingerprint({}, "checkpoint", expected)
    with pytest.raises(RuntimeError, match="missing or mismatched"):
        protocol_fingerprint.verify_fingerprint(
            {"protocol_fingerprint": {"sha256": "different"}}, "checkpoint", expected)


@pytest.mark.parametrize('family,adapter,proposal,prompt_group', [
    ('yolo26_p2', 'ultralytics_yolo26_p2', 'fred_yolo26_proposals', 'FRED_YOLO26_P2'),
    ('dfine', 'dfine', 'fred_dfine_proposals', 'FRED_DFINE'),
])
def test_custom_detector_resume_detects_prompt_or_operator_change(
    tmp_path, monkeypatch, family, adapter, proposal, prompt_group,
):
    fred = tmp_path / 'sota/FRED_LLM_GE'
    protocol = fred / 'configs/phase1/protocol.yaml'
    protocol.parent.mkdir(parents=True)
    protocol.write_text(yaml.safe_dump({
        'families': {family: {'data': 'challenging_search'}},
        'challenging': {'data_root': 'cache',
                        'search': {'train_frame_step': 50, 'val_frame_step': 10}},
    }))
    seed = fred / 'seeds/seed.py'
    paths = [seed, fred / 'cache/splits/challenging_train_split.txt',
             fred / 'cache/search_train_step50/train.txt',
             fred / 'cache/search_val_step10/train.txt',
             tmp_path / 'run_improved.py',
             tmp_path / 'src/cfg/constants_fred_yolo11.py',
             tmp_path / 'src/llm_utils.py',
             fred / 'phase1_detection/candidate_runner.py',
             fred / 'phase1_detection/export_split.py',
             fred / 'phase1_detection/fitness.py',
             fred / 'phase1_detection/map_reporter.py',
             fred / f'phase1_detection/adapters/{adapter}.py',
             tmp_path / 'src/llm_mutation.py', tmp_path / 'src/llm_crossover.py',
             tmp_path / f'src/{proposal}.py', tmp_path / 'slurm-config/slurm_config.yaml',
             tmp_path / f'templates/{prompt_group}/Normal/1.txt',
             tmp_path / 'templates/CrossOver/crossover.txt']
    if family == 'dfine':
        paths.extend([tmp_path / 'src/fred_dfine_genome.py',
                      tmp_path / 'src/fred_dfine_selection.py'])
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('first\n')
    monkeypatch.setattr(protocol_fingerprint.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout='commit-1\n'))
    monkeypatch.setattr(protocol_fingerprint.importlib.metadata, 'version',
                        lambda _: 'test-version')
    first = protocol_fingerprint.compute_fingerprint(tmp_path, family, 'tag', seed)
    assert 'prompt:templates/CrossOver/crossover.txt' in first['files']
    prompt = tmp_path / f'templates/{prompt_group}/Normal/1.txt'
    prompt.write_text('second\n')
    assert first['sha256'] != protocol_fingerprint.compute_fingerprint(
        tmp_path, family, 'tag', seed)['sha256']
    if family == 'dfine':
        prompt.write_text('first\n')
        selection = tmp_path / 'src/fred_dfine_selection.py'
        selection.write_text('second\n')
        assert first['sha256'] != protocol_fingerprint.compute_fingerprint(
            tmp_path, family, 'tag', seed)['sha256']
