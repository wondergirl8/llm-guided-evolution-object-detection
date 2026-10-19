import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.validator import SEED, validate_source, validate_genes
from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn import faster_rcnn_prediction, faster_rcnn_frame

class Array:
    def __init__(self, values): self.values = values
    def tolist(self): return self.values

def result(boxes=None, labels=None, scores=None):
    return SimpleNamespace(metainfo={"ori_shape": (100, 200)}, pred_instances=SimpleNamespace(
        bboxes=Array([[10., 20., 30., 40.]] if boxes is None else boxes),
        labels=Array([0] if labels is None else labels),
        scores=Array([0.8] if scores is None else scores)))

def test_seed_and_all_prototype_choices():
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.validator import SCHEMA
    genes = validate_source(SEED)
    for fields in SCHEMA["groups"].values():
        for field, choices in fields.items():
            for choice in choices:
                modified = copy.deepcopy(genes)
                modified[field] = choice
                assert validate_genes(modified) == modified

@pytest.mark.parametrize("replacement", [
    'class FeatureGenes:\n    values = {"fpn_channels": 512}',
    'class FeatureGenes:\n    values = {"fpn_channels": True}',
    'class FeatureGenes:\n    values = dict(fpn_channels=256)',
    'class FeatureGenes:\n    values = {"fpn_channels": 256}\n    def run(self): pass',
    '@print\nclass FeatureGenes:\n    values = {"fpn_channels": 256}',
])
def test_rejects_unsafe_or_illegal_chunks(tmp_path, replacement):
    parts = SEED.read_text().split("# --OPTION--")
    parts[1] = "\n" + replacement + "\n"
    path = tmp_path / "candidate.py"
    path.write_text("# --OPTION--".join(parts))
    with pytest.raises((ValueError, SyntaxError)):
        validate_source(path)

def test_rejects_protected_changes(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("import os\n" + SEED.read_text())
    with pytest.raises(ValueError): validate_source(path)

def test_converts_original_pixel_prediction():
    target = {"boxes": [[10., 20., 30., 40.]], "labels": [0]}
    frame = faster_rcnn_frame(result(), sample_id="sequence:1", target=target)
    assert frame["prediction"] == {"boxes": target["boxes"], "labels": [0], "scores": [0.8]}
    assert frame["width"] == 200 and frame["height"] == 100
    assert frame["target"] is target

def test_empty_predictions():
    assert faster_rcnn_prediction(result([], [], [])) == {"boxes": [], "labels": [], "scores": []}

@pytest.mark.parametrize("bad", [result(labels=[1]), result(scores=[float("nan")]),
    result(scores=[1.1]), result(boxes=[[0, 0, 0, 1]]), result(labels=[]),
    result(boxes=[[0, 0, float("inf"), 1]])])
def test_rejects_malformed_predictions(bad):
    with pytest.raises(ValueError): faster_rcnn_prediction(bad)

def test_rejects_unrestored_coordinates():
    with pytest.raises(ValueError):
        faster_rcnn_frame(result(boxes=[[0, 0, 201, 20]]), sample_id="a", target={"boxes": [], "labels": []})

def test_defaults_do_not_change_yolo_configuration(monkeypatch):
    import importlib
    import sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.validator as validator
    from src.cfg import constants
    monkeypatch.setenv("LLMGE_DETECTOR", "faster_rcnn")
    importlib.reload(constants)
    assert constants.num_generations == 3
    assert constants.start_population_size == 4
    assert "faster_rcnn" in constants.TRAIN_FILE
    monkeypatch.setenv("LLMGE_DETECTOR", "yolo11")
    importlib.reload(constants)
    assert "yolo11" in constants.TRAIN_FILE


def test_yolo_uses_shared_data_loader_without_redefining_it():
    import ast
    path = "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_eval.py"
    tree=ast.parse(Path(path).read_text()); local={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert not {"load_annotations","load_sequence","load_split"}&local
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=="event_data"]
    assert len(imports)==1 and {"load_annotations","load_sequence","load_split"}<={a.name for a in imports[0].names}


def test_mmdetection_alternative_genome_and_geometry():
    pytest.importorskip('mmdet')
    pytest.importorskip('mmcv.ops')
    import torch
    from mmengine.logging import MMLogger
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.network import build_config, build_detector
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.train_eval import prepare_batch, synthetic_data, preflight
    checkpoint = Path('sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn/checkpoints/coco.pth')
    if not checkpoint.exists():
        pytest.skip('Explicit checkpoint setup required')
    MMLogger.get_current_instance().setLevel('WARNING')
    genes = validate_source(SEED)
    genes.update(fpn_channels=128, rpn_channels=128, rpn_convs=2,
                 anchor_scales=[4], anchor_ratios=[1.0], roi_size=14,
                 roi_sampling_ratio=2, box_fc_channels=512)
    model, transfer = build_detector(build_config(genes), checkpoint)
    assert 'roi_head.bbox_head.fc_cls.weight' in transfer['skipped']
    assert 'backbone.conv1.weight' in transfer['loaded']
    batch = prepare_batch(model, synthetic_data(), [0, 1], False)
    assert batch['inputs'].shape == (2, 3, 64, 96)
    assert batch['data_samples'][0].metainfo['ori_shape'] == (128, 192)
    assert batch['data_samples'][0].metainfo['scale_factor'] == (0.5, 0.5)
    assert batch['data_samples'][1].gt_instances.bboxes.shape == (0, 4)
    preflight(model)


def test_invalid_candidate_records_failure_before_execution(tmp_path, monkeypatch):
    pytest.importorskip('datasets')
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn import train_eval as runner
    source = SEED.read_text().replace('"fpn_channels": 256', '"fpn_channels": 999')
    (tmp_path / 'network_broken.py').write_text(source)
    monkeypatch.setattr(runner, 'SCRIPT_DIR', tmp_path)
    args = SimpleNamespace(model='network_broken', variant_dir=str(tmp_path), smoke=False, seed=0)
    with pytest.raises(ValueError):
        runner.run(args)
    import json
    status = json.loads((tmp_path / 'reports/broken/status.json').read_text())
    assert status['status'] == 'failed'
    assert status['stage'] == 'candidate_validation'
    assert (tmp_path / 'results/broken_results.csv').read_text().strip() == '0.000000,0.000000,999999999'
    assert not (tmp_path / 'trained_models').exists()
