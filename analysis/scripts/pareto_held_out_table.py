"""Combine Pareto challenging-test reports into CSV and Markdown tables."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; base=ROOT/"sota/FRED_LLM_GE/phase1_detection/seeds/faster_rcnn_scratch/pareto_held_out_comparison"; reports=base/"reports/held_out_reports"
rows=[]
for path in sorted(reports.glob("*_evaluation.json")):
    report=json.loads(path.read_text()); metrics=report["metrics"]
    rows.append({"individual":report["gene_id"],"mAP50":metrics["mAP50"],"mAP50_95":metrics["mAP50_95"],"params":metrics["params"],"frames":report["num_frames"],"report":str(path.relative_to(ROOT))})
if len(rows)!=3: raise SystemExit(f"Expected 3 held-out reports, found {len(rows)}")
base.mkdir(parents=True,exist_ok=True)
import csv
with (base/"challenging_test_results.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
lines=["# Pareto-model challenging_test comparison","","> Comparative held-out study: all three Pareto architectures were evaluated on the official test split.","","| Individual | mAP50 | mAP50:95 | Parameters | Frames |","|---|---:|---:|---:|---:|"]
for row in sorted(rows,key=lambda value:value["mAP50_95"],reverse=True): lines.append(f"| `{row['individual']}` | {row['mAP50']*100:.2f}% | {row['mAP50_95']*100:.2f}% | {row['params']:,} | {row['frames']:,} |")
(base/"challenging_test_results.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))
