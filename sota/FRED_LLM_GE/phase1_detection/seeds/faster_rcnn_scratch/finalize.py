"""Freeze generation-20 winner, retrain from scratch, then evaluate held-out test."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import torch
SCRIPT_DIR=Path(__file__).resolve().parent; ROOT=SCRIPT_DIR.parents[4]; FRED_DIR=SCRIPT_DIR.parents[2]
for p in (ROOT,SCRIPT_DIR.parents[1],FRED_DIR/"data",FRED_DIR/"phase0_data"): sys.path.insert(0,str(p))
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_detector
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.official_split import TEST_SPLIT_SHA256,official_test_archives
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import file_sha256,freeze_manifest,load_generation,select_winner
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.train_eval import load_variant,predict_frames,preflight,set_random_seed,train,write_json
from event_data import FRAME_STEP,TRAIN_SEQUENCES,VAL_SEQUENCES,load_archives,load_split
from fred_stream import REVISION
import map_reporter
def candidate_source(gene_id):
    return (FRED_DIR/"seeds/seed_faster_rcnn_scratch.py","network") if gene_id=="seed" else (SCRIPT_DIR/"models"/f"network_{gene_id}.py",f"network_{gene_id}")
def run(args):
    checkpoint,archive,records=load_generation(args.checkpoints,20); gene_id,fitness=select_winner(records); source,model_name=candidate_source(gene_id)
    if not source.is_file(): raise FileNotFoundError(f"Selected candidate source is missing: {source}")
    _,genes,_=load_variant(model_name,SCRIPT_DIR/"models")
    selection={"schema_version":"fred-faster-rcnn-scratch-winner-v1","generation":20,"gene_id":gene_id,"selection_rule":["max_validation_mAP50_95","max_validation_mAP50","min_parameters"],"validation_fitness":{"mAP50":fitness[0],"mAP50_95":fitness[1],"params":int(fitness[2])},"genes":genes,"source_file":str(source.relative_to(ROOT)),"source_sha256":file_sha256(source),"generation_checkpoint":str(checkpoint.relative_to(ROOT)),"generation_checkpoint_sha256":file_sha256(checkpoint),"ancestry":archive.get("GLOBAL_DATA_ANCESTRY",{}).get(gene_id,{})}
    output=SCRIPT_DIR/"final_results"; freeze_manifest(output/"winner_selection.json",selection)
    report_path=SCRIPT_DIR/"held_out_reports"/f"{gene_id}_evaluation.json"
    if report_path.exists(): raise FileExistsError(f"Held-out evaluation already exists: {report_path}")
    set_random_seed(args.seed); cfg,_,_=load_variant(model_name,SCRIPT_DIR/"models"); model,initialization=build_detector(cfg,seed=args.seed); model.to(args.device); preflight(model)
    sequences=(TRAIN_SEQUENCES+","+VAL_SEQUENCES).split(","); training={"epochs":20,"batch_size":args.batch_size,"lr":args.lr,"seed":0,"frame_step":10,"img_width":640,"train_sequences":sequences}; weights=output/"winner_20epoch.pt"
    if weights.exists():
        saved=torch.load(weights,map_location="cpu")
        if saved.get("selection")!=selection or saved.get("training")!=training: raise ValueError("Existing final checkpoint differs from frozen protocol")
        model.load_state_dict(saved["state_dict"],strict=True); training_frames=saved["training_frames"]
    else:
        data=load_split("train",sequences,10,640); train(model,data,SimpleNamespace(epochs=20,batch_size=args.batch_size,lr=args.lr)); training_frames=len(data["images"])
        tmp=weights.with_suffix(f".{os.getpid()}.tmp"); torch.save({"state_dict":model.state_dict(),"genes":genes,"selection":selection,"initialization":initialization,"training":training,"training_frames":training_frames},tmp); os.replace(tmp,weights)
    test_archives=official_test_archives(); test_sequences=[sequence for _,sequence in test_archives]; test=load_archives(test_archives,10,640); frames=predict_frames(model,test,args.batch_size); write_json(output/"held_out_predictions.json",frames)
    metadata={"dataset_revision":REVISION,"manifest_identity":TEST_SPLIT_SHA256,"split_identity":"official_challenging_test:"+",".join(test_sequences),"annotation_policy":"fred_coordinates_txt_clipped_to_image_trusted_evaluator","model_identity":selection["source_sha256"],"checkpoint_identity":file_sha256(weights),"code_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"postprocessing":"MMDetection rescale=True score=0.05 rcnn_nms=0.5 max_per_img=100","training_config_identity":hashlib.sha256(json.dumps(training,sort_keys=True).encode()).hexdigest(),"modality":"event","benchmark_split":"challenging","project_split":"held_out_test","official_split":"challenging_test","initialization_policy":initialization["policy"],"initial_state_sha256":initialization["initial_state_sha256"]}
    result,report=map_reporter.report({"metadata":metadata,"frames":frames,"params":sum(p.numel() for p in model.parameters()),"expected_sample_ids":test["sample_ids"]},sota_root=SCRIPT_DIR,gene_id=gene_id,mode="held-out")
    write_json(output/"final_run_manifest.json",{"selection":selection,"initialization":initialization,"training":training,"training_frames":training_frames,"held_out_sequences":test_sequences,"held_out_frames":len(frames),"weights":str(weights.relative_to(ROOT)),"weights_sha256":file_sha256(weights),"held_out_report":str(report.relative_to(ROOT)),"metrics":vars(result)})
def cli():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoints",default=str(ROOT/"fred_faster_rcnn_scratch_checkpoints")); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch_size",type=int,default=2); p.add_argument("--frame_step",type=int,default=FRAME_STEP); p.add_argument("--img_width",type=int,default=640); p.add_argument("--lr",type=float,default=.0025); p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda"); args=p.parse_args()
    if (args.epochs,args.frame_step,args.img_width,args.seed)!=(20,10,640,0): p.error("Frozen final protocol requires epochs=20, frame_step=10, img_width=640, seed=0")
    run(args)
if __name__=="__main__": cli()
