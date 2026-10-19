"""Plot validation learning curves and the accuracy-versus-epoch Pareto front."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def pareto_mask(df):
    mask=[]
    for _,row in df.iterrows():
        dominates=(df.mAP50>=row.mAP50)&(df.mAP50_95>=row.mAP50_95)&(df.epoch<=row.epoch)
        strict=(df.mAP50>row.mAP50)|(df.mAP50_95>row.mAP50_95)|(df.epoch<row.epoch)
        mask.append(not (dominates&strict).any())
    return pd.Series(mask,index=df.index)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--history",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--title",default="Faster R-CNN scratch: validation performance by epoch"); args=p.parse_args()
    df=pd.DataFrame(json.loads(args.history.read_text())).sort_values("epoch"); df["pareto_front"]=pareto_mask(df); args.output.mkdir(parents=True,exist_ok=True); df.to_csv(args.output/"epoch_metrics.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(df.epoch,df.mAP50*100,marker="o",label="mAP50"); ax.plot(df.epoch,df.mAP50_95*100,marker="o",label="mAP50:95"); ax.set(xlabel="Training epoch",ylabel="Validation score (%)",title=args.title); ax.set_xticks(df.epoch); ax.grid(ls=":",alpha=.6); ax.legend(); fig.tight_layout(); fig.savefig(args.output/"performance_by_epoch.png",dpi=220); plt.close(fig)
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(projection="3d"); other=df[~df.pareto_front]; front=df[df.pareto_front]; ax.scatter(other.epoch,other.mAP50*100,other.mAP50_95*100,color="lightgray",label="Dominated epochs"); ax.plot(front.epoch,front.mAP50*100,front.mAP50_95*100,color="crimson",marker="o",label="Pareto front"); ax.set(xlabel="Epochs (training cost)",ylabel="mAP50 (%)",zlabel="mAP50:95 (%)",title=args.title+"\nPareto front: accuracy versus training epochs"); ax.legend(); fig.tight_layout(); fig.savefig(args.output/"epoch_pareto_front.png",dpi=220); plt.close(fig)
    print(f"Wrote {len(df)} epoch measurements and graphs to {args.output}")
if __name__=="__main__": main()
