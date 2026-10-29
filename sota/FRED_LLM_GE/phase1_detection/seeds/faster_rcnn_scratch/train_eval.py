"""Train and validate one randomly initialized Faster R-CNN candidate."""
import argparse, hashlib, importlib.metadata, importlib.util, json, os, random, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
SCRIPT_DIR=Path(__file__).resolve().parent; ROOT=SCRIPT_DIR.parents[4]; FRED_DIR=SCRIPT_DIR.parents[2]
os.environ.setdefault("HF_HOME",str(ROOT/".cache-faster-rcnn/huggingface"))
for p in (ROOT,SCRIPT_DIR.parents[1],FRED_DIR/"data",FRED_DIR/"phase0_data"): sys.path.insert(0,str(p))
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.network import build_detector
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.validator import SEED,validate_source
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.train_eval import prepare_batch,predict_frames,preflight,synthetic_data,train
from fitness import evaluate_frames
from event_data import FRAME_STEP,TRAIN_SEQUENCES,VAL_SEQUENCES,load_split
from fred_stream import REVISION
import map_reporter
from result_contract import DetectionResult,validate_gene_id,write_results

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)); os.replace(tmp,path)
def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def load_variant(name,directory):
    path=SEED if name=="network" else Path(directory)/f"{name}.py"; genes=validate_source(path)
    spec=importlib.util.spec_from_file_location("fred_frcnn_scratch_candidate",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_model_config(),genes,path
def train_with_epoch_validation(model,train_data,val_data,args,params,run_dir):
    """Train continuously and measure the same validation split after each epoch."""
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9,weight_decay=.0001); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); order=torch.randperm(len(train_data["images"])).tolist(); loss=None
        for start in range(0,len(order),args.batch_size):
            batch=prepare_batch(model,train_data,order[start:start+args.batch_size],True); optimizer.zero_grad()
            loss,_=model.parse_losses(model(**batch,mode="loss"))
            if not torch.isfinite(loss): raise ValueError("Non-finite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError("Non-finite gradients")
            optimizer.step()
        frames=predict_frames(model,val_data,args.batch_size)
        result=evaluate_frames(frames,params=params,expected_sample_ids=val_data["sample_ids"])
        row={"epoch":epoch,"final_batch_loss":float(loss.item()),"mAP50":float(result.map50),"mAP50_95":float(result.map50_95),"params":int(params)}
        history.append(row); write_json(run_dir/"epoch_history.json",history)
        print(f"epoch {epoch}/{args.epochs}: loss={loss.item():.5f}, mAP50={result.map50:.6f}, mAP50:95={result.map50_95:.6f}",flush=True)
    return history
def run(args):
    gene_id="smoke" if args.smoke else (args.run_name or ("seed" if args.model=="network" else args.model.removeprefix("network_"))); validate_gene_id(gene_id)
    run_dir=SCRIPT_DIR/"reports"/gene_id; set_random_seed(args.seed); config=vars(args).copy(); start=time.time(); stage="candidate_validation"
    try:
        cfg,genes,path=load_variant(args.model,args.variant_dir); run_dir.mkdir(parents=True,exist_ok=True); (run_dir/"candidate_source.py").write_text(path.read_text())
        sources=[Path(__file__),SCRIPT_DIR/"network.py",SCRIPT_DIR/"validator.py",SCRIPT_DIR.parents[1]/"event_data.py"]
        write_json(run_dir/"code_identity.json",{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
        stage="model_initialization"; model,initialization=build_detector(cfg,seed=args.seed); model.to(args.device)
        if str(args.device).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
        write_json(run_dir/"initialization.json",initialization)
        write_json(run_dir/"resolved_config.json",{"model":cfg.to_dict(),"training":config,"genes":genes,"versions":{n:importlib.metadata.version(n) for n in ("torch","torchvision","mmcv","mmengine","mmdet","datasets","pycocotools")}})
        params=sum(p.numel() for p in model.parameters()); stage="model_preflight"; preflight(model); stage="data_loading"
        if args.smoke: train_data=val_data=synthetic_data(); args.epochs=1
        else:
            train_data=load_split("train",args.train_sequences.split(","),args.frame_step,args.img_width)
            val_data=load_split("train",args.val_sequences.split(","),args.frame_step,args.img_width)
        stage="training"
        history=train_with_epoch_validation(model,train_data,val_data,args,params,run_dir) if args.track_epochs and not args.smoke else None
        if history is None: train(model,train_data,args)
        model_path=SCRIPT_DIR/"trained_models"/f"{gene_id}.pt"; model_path.parent.mkdir(parents=True,exist_ok=True); tmp=model_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"state_dict":model.state_dict(),"config":config,"genes":genes,"initialization":initialization},tmp); os.replace(tmp,model_path)
        stage="evaluation"; frames=predict_frames(model,val_data,args.batch_size)
        if args.smoke:
            write_json(run_dir/"smoke.json",{"status":"passed","device":args.device,"frames":frames,"params":params,"initialization":initialization}); write_json(run_dir/"status.json",{"status":"success","mode":"smoke"}); return
        metadata={"dataset_revision":REVISION,"manifest_identity":hashlib.sha256(json.dumps({k:config[k] for k in ("train_sequences","val_sequences","frame_step")},sort_keys=True).encode()).hexdigest(),"split_identity":f"train={args.train_sequences};val={args.val_sequences}","annotation_policy":"fred_coordinates_txt_clipped_to_image","model_identity":hashlib.sha256(path.read_bytes()).hexdigest(),"checkpoint_identity":hashlib.sha256(model_path.read_bytes()).hexdigest(),"code_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"postprocessing":"MMDetection rescale=True score=0.05 rcnn_nms=0.5 max_per_img=100","training_config_identity":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),"modality":"event","benchmark_split":"challenging","project_split":"validation","official_split":"challenging_train","initialization_policy":initialization["policy"],"initial_state_sha256":initialization["initial_state_sha256"]}
        result,report=map_reporter.report({"metadata":metadata,"frames":frames,"params":params,"expected_sample_ids":val_data["sample_ids"]},sota_root=SCRIPT_DIR,gene_id=gene_id,mode="validation")
        write_json(run_dir/"status.json",{"status":"success","seconds":time.time()-start,"fitness":vars(result),"report":str(report),"training_frames":len(train_data["images"]),"validation_frames":len(frames),"epoch_history":str(run_dir/"epoch_history.json") if history else None})
        print(f"mAP50={result.map50}, mAP50:95={result.map50_95}, params={params}",flush=True)
    except Exception as error:
        write_json(run_dir/"status.json",{"status":"failed","stage":stage,"error":repr(error),"traceback":traceback.format_exc(),"training":config})
        if not args.smoke: write_results(SCRIPT_DIR,gene_id,DetectionResult(0.0,0.0,999999999))
        raise
    finally: print("job done",flush=True)
def cli():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",default="network"); p.add_argument("--variant_dir",default=str(SCRIPT_DIR/"models")); p.add_argument("--train_sequences",default=TRAIN_SEQUENCES); p.add_argument("--val_sequences",default=VAL_SEQUENCES); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch_size",type=int,default=2); p.add_argument("--frame_step",type=int,default=FRAME_STEP); p.add_argument("--img_width",type=int,default=640); p.add_argument("--lr",type=float,default=.0025); p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda"); p.add_argument("--smoke",action="store_true"); p.add_argument("--track_epochs",action="store_true"); p.add_argument("--run_name"); args=p.parse_args()
    if args.run_name: validate_gene_id(args.run_name)
    for name in ("train_sequences","val_sequences"):
        values=[v.strip() for v in getattr(args,name).split(",")]
        if not values or any(not v.isdigit() for v in values) or len(values)!=len(set(values)): p.error(f"{name} requires distinct numeric sequence IDs")
        setattr(args,name,",".join(values))
    if set(args.train_sequences.split(","))&set(args.val_sequences.split(",")): p.error("Training and validation sequences overlap")
    if min(args.epochs,args.batch_size,args.frame_step)<=0 or args.img_width<32 or not np.isfinite(args.lr) or args.lr<=0: p.error("Invalid training budget or learning rate")
    run(args)
if __name__=="__main__": cli()
