import copy,inspect,pickle
from pathlib import Path
import pytest
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.validator import SCHEMA,SEED,validate_genes,validate_source
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import freeze_manifest,load_generation,select_winner

def test_seed_and_choices():
    genes=validate_source(SEED)
    for fields in SCHEMA["groups"].values():
        for field,choices in fields.items():
            for choice in choices:
                candidate=copy.deepcopy(genes); candidate[field]=choice; assert validate_genes(candidate)==candidate
def test_seed_uses_scratch_builder_and_no_checkpoint():
    source=SEED.read_text(); assert "faster_rcnn_scratch.network" in source and "checkpoint" not in source.lower()
def test_twenty_generations(monkeypatch):
    import importlib
    from src.cfg import constants
    monkeypatch.setenv("LLMGE_DETECTOR","faster_rcnn_scratch"); importlib.reload(constants)
    assert list(range(1,constants.num_generations))==list(range(1,21)); assert constants.population_size==4 and constants.num_elites==2
@pytest.mark.parametrize("records,winner",[
    ({"a":{"fitness":(.8,.4,10)},"b":{"fitness":(.7,.5,20)}},"b"),
    ({"a":{"fitness":(.8,.5,30)},"b":{"fitness":(.9,.5,40)}},"b"),
    ({"a":{"fitness":(.9,.5,30)},"b":{"fitness":(.9,.5,20)}},"b")])
def test_winner_ties(records,winner): assert select_winner(records)[0]==winner
def test_requires_gen20(tmp_path):
    with pytest.raises(FileNotFoundError): load_generation(tmp_path)
    with (tmp_path/"global_gen_20.pkl").open("wb") as f: pickle.dump({"GLOBAL_DATA":{"a":{"fitness":(.1,.1,1)}}},f)
    assert load_generation(tmp_path)[2]["a"]["fitness"]==(.1,.1,1)
def test_manifest_immutable(tmp_path):
    path=tmp_path/"winner.json"; freeze_manifest(path,{"gene_id":"a"}); freeze_manifest(path,{"gene_id":"a"})
    with pytest.raises(ValueError): freeze_manifest(path,{"gene_id":"b"})
def test_no_checkpoint_builder_api():
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_detector
    assert list(inspect.signature(build_detector).parameters)==["cfg","seed"]
def test_deterministic_initialization():
    pytest.importorskip("mmdet"); pytest.importorskip("mmcv.ops")
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_config,build_detector
    cfg=build_config(validate_source(SEED)); first,a=build_detector(cfg,seed=0); assert all(getattr(m,"init_cfg",None) is None for m in first.modules()); del first
    second,b=build_detector(cfg,seed=0); del second; third,c=build_detector(cfg,seed=1); del third
    assert a["pretrained"] is False and a["checkpoint"] is None and a["initial_state_sha256"]==b["initial_state_sha256"]!=c["initial_state_sha256"]
def test_official_parser():
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.official_split import parse_official_split
    assert parse_official_split("9/\n10/\n")==["9","10"]
    with pytest.raises(ValueError): parse_official_split("test/9/\n")
def test_final_contract_source():
    source=Path("sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/finalize.py").read_text()
    assert 'mode="held-out"' in source and "winner_selection.json" in source and 'TRAIN_SEQUENCES+","+VAL_SEQUENCES' in source
