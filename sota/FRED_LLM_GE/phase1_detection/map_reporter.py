""""
Evaluates one individual candidate model and creates a report for that individual.
write_results(sota_root, gene_id, result) -> create the csv report for run_improved.py
report() -> creates a json file for humans to read
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

try:
    from .fitness import evaluate_frames
    from .result_contract import validate_gene_id, write_results
except ImportError:
    from fitness import evaluate_frames
    from result_contract import validate_gene_id, write_results


""""
payload: dictionary with keys "metadata", "frames", "params" 
sota_root: output directory
gene_id: need the candidate's id to be providid os we can create and evaluation 
file for each, and the id will be included in the file name
""""
def report(payload, *, sota_root, gene_id, mode="validation"):
    """Validation CSV feeds the legacy orchestrator; held-out JSON never does.

    The caller is responsible for enforcing trusted-evaluator label access.
    Split declarations are provenance, not an access-control mechanism.
    """
    validate_gene_id(gene_id)
    if mode not in ("validation", "held-out"):
        raise ValueError("mode must be validation or held-out")


    # make sure all the required filed are provided
    # derived the required fields from [MASTER_PROJECT_PLAN.md (line 1034)]
    metadata = payload["metadata"]
    required = ("dataset_revision", "manifest_identity", "split_identity",
                "annotation_policy", "model_identity", "checkpoint_identity",
                "code_revision", "postprocessing", "training_config_identity")
    for key in required:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata requires a nonempty {key}")


    # make sure we have coorent modality, split, 
    if metadata.get("modality") not in ("rgb", "event", "rgb_event"):
    raise ValueError("metadata requires modality rgb, event or rgb_event")
    if metadata.get("benchmark_split") not in ("canonical", "challenging"):
        raise ValueError("benchmark_split must be canonical or challenging")
    expected_partition = "validation" if mode == "validation" else "held_out_test"
    if metadata.get("project_split") != expected_partition:
        raise ValueError(f"{mode} mode requires project_split={expected_partition}")
    if mode == "validation" and metadata.get("official_split") != "challenging_train":
        raise ValueError("search validation must come from official challenging_train")
    if mode == "held-out" and not metadata.get("official_split"):
        raise ValueError("held-out reporting requires an official_split declaration")


    # do evaluation 

    result = evaluate_frames(payload["frames"], params=payload["params"],
                         expected_sample_ids=payload["expected_sample_ids"])
    directory = Path(sota_root) / ("results" if mode == "validation" else "held_out_reports")
    directory.mkdir(parents=True, exist_ok=True)

    # create summery
    summary = {
    "schema_version": "fred-detection-report-v1", "gene_id": gene_id,
    "mode": mode, "metadata": metadata,
    "metrics": {"mAP50": result.map50, "mAP50_95": result.map50_95,
                "params": result.params},
    "metric_units": "fraction", "num_frames": len(payload["frames"]),
    "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True,
                          allow_nan=False).encode()).hexdigest(),
    "evaluator": {"backend": "pycocotools",
                  "version": importlib.metadata.version("pycocotools"),
                  "iou_type": "bbox", "iou_thresholds": [0.5+i*0.05 for i in range(10)],
                  "recall_threshold_count": 101, "max_detections": [1, 10, 100]},
    "comparison_status": "metric-compatible; exact paper reproduction not established",
    }

    #  write the JSON file:
    output = directory / f"{gene_id}_evaluation.json"
    from tempfile import NamedTemporaryFile
    import os
    name = None

    try:
        with NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
            name = handle.name
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, output)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)




    # calling result_contract.py to write a cvs report
    if mode == "validation":
        write_results(sota_root, gene_id, result)
    return result, output



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sota-root", required=True, type=Path)
    parser.add_argument("--gene-id", required=True)
    parser.add_argument("--mode", choices=("validation", "held-out"), default="validation")
    args = parser.parse_args()
    result, output = report(json.loads(args.input.read_text()), sota_root=args.sota_root,
                            gene_id=args.gene_id, mode=args.mode)
    print(f"mAP50={result.map50:.6f}, mAP50:95={result.map50_95:.6f}, params={result.params}")
    print(output)
    if args.mode == "validation":
        print("job done", flush=True)


if __name__ == "__main__":
    main()