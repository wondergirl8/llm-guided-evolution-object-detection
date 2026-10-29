"""Continue one Pareto candidate for 17 epochs, then evaluate challenging_test."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import torch

SCRIPT_DIR=Path(__file__).resolve().parent; ROOT=SCRIPT_DIR.parents[4]; FRED_DIR=SCRIPT_DIR.parents[2]
for path in (ROOT,SCRIPT_DIR.parents[1],FRED_DIR/"data",FRED_DIR/"phase0_data"): sys.path.insert(0,str(path))
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_detector
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.official_split import TEST_SPLIT_SHA256,official_test_archives
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import file_sha256
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.train_eval import load_variant,predict_frames,set_random_seed,train,write_json
from event_data import TRAIN_SEQUENCES,VAL_SEQUENCES,load_archives,load_split
from fred_stream import REVISION
import map_reporter

PARETO_IDS={"xXx005EzIza8AimNURDwEWdbye4","xXxT2dgPIjQdOfJ2sLEswEb2hTV","xXx7KjH1KlxilhpYvHIVxTndgfe"}

def run(args):
    if args.gene_id not in PARETO_IDS: raise ValueError("gene_id is not in the frozen Pareto comparison set")
    model_name=f"network_{args.gene_id}"; source=SCRIPT_DIR/"models"/f"{model_name}.py"; search_weights=SCRIPT_DIR/"trained_models"/f"{args.gene_id}.pt"
    if not source.is_file() or not search_weights.is_file(): raise FileNotFoundError("Candidate source or three-epoch checkpoint is missing")
    output=SCRIPT_DIR/"pareto_held_out_comparison"; weights_dir=output/"weights"; predictions_dir=output/"predictions"; reports_root=output/"reports"
    for directory in (weights_dir,predictions_dir,reports_root): directory.mkdir(parents=True,exist_ok=True)
    final_weights=weights_dir/f"{args.gene_id}_continued_17epoch.pt"; report_path=reports_root/"held_out_reports"/f"{args.gene_id}_evaluation.json"
    if report_path.exists(): raise FileExistsError(f"Comparison report already exists: {report_path}")
    set_random_seed(0); cfg,genes,_=load_variant(model_name,SCRIPT_DIR/"models"); model,initialization=build_detector(cfg,seed=0); model.to(args.device)
    saved=torch.load(search_weights,map_location="cpu",weights_only=False)
    sequences=(TRAIN_SEQUENCES+","+VAL_SEQUENCES).split(",")
    continuation={"initial_search_epochs":3,"continued_epochs":17,"total_epochs":20,"optimizer_policy":"new_SGD_optimizer_for_continuation","batch_size":2,"lr":.0025,"momentum":.9,"weight_decay":.0001,"seed":0,"frame_step":10,"img_width":640,"train_sequences":sequences}
    if final_weights.exists():
        continued=torch.load(final_weights,map_location="cpu",weights_only=False)
        if continued.get("continuation")!=continuation or continued.get("source_search_checkpoint_sha256")!=file_sha256(search_weights): raise ValueError("Saved continuation checkpoint provenance mismatch")
        model.load_state_dict(continued["state_dict"],strict=True); training_frames=continued.get("training_frames")
    else:
        model.load_state_dict(saved["state_dict"],strict=True); train_data=load_split("train",sequences,10,640); training_frames=len(train_data["images"]); train(model,train_data,SimpleNamespace(epochs=17,batch_size=2,lr=.0025))
        tmp=final_weights.with_suffix(f".{os.getpid()}.tmp"); torch.save({"state_dict":model.state_dict(),"genes":genes,"initialization":initialization,"continuation":continuation,"training_frames":training_frames,"source_search_checkpoint":str(search_weights.relative_to(ROOT)),"source_search_checkpoint_sha256":file_sha256(search_weights)},tmp); os.replace(tmp,final_weights)
    test_archives=official_test_archives(); test_sequences=[sequence for _,sequence in test_archives]; test=load_archives(test_archives,10,640); frames=predict_frames(model,test,2); write_json(predictions_dir/f"{args.gene_id}.json",frames)
    metadata={"dataset_revision":REVISION,"manifest_identity":TEST_SPLIT_SHA256,"split_identity":"official_challenging_test:"+",".join(f"{split}/{sequence}" for split,sequence in test_archives),"annotation_policy":"fred_coordinates_txt_clipped_to_image_trusted_evaluator","model_identity":file_sha256(source),"checkpoint_identity":file_sha256(final_weights),"code_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"postprocessing":"MMDetection rescale=True score=0.05 rcnn_nms=0.5 max_per_img=100","training_config_identity":hashlib.sha256(json.dumps(continuation,sort_keys=True).encode()).hexdigest(),"modality":"event","benchmark_split":"challenging","project_split":"held_out_test","official_split":"challenging_test","study_design":"comparative_pareto_held_out_evaluation","initialization_policy":saved.get("initialization",initialization).get("policy","unknown"),"source_search_checkpoint_sha256":file_sha256(search_weights)}
    result,report=map_reporter.report({"metadata":metadata,"frames":frames,"params":sum(p.numel() for p in model.parameters()),"expected_sample_ids":test["sample_ids"]},sota_root=reports_root,gene_id=args.gene_id,mode="held-out")
    write_json(output/f"{args.gene_id}_manifest.json",{"gene_id":args.gene_id,"continuation":continuation,"training_frames":training_frames,"held_out_archives":test_archives,"held_out_frames":len(frames),"source_weights":str(search_weights.relative_to(ROOT)),"final_weights":str(final_weights.relative_to(ROOT)),"report":str(report.relative_to(ROOT)),"metrics":vars(result)})
    print(f"mAP50={result.map50}, mAP50:95={result.map50_95}, params={result.params}",flush=True)

def cli():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--gene_id",required=True); parser.add_argument("--device",default="cuda"); args=parser.parse_args(); run(args)
if __name__=="__main__": cli()
