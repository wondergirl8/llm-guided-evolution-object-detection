"""Generation-20 archive loading and deterministic winner selection."""
import hashlib, json, pickle
from pathlib import Path
FAILED_PARAMS = 999_999_999
def valid_fitness(v):
    return isinstance(v,(tuple,list)) and len(v)==3 and 0<=v[0]<=1 and 0<=v[1]<=1 and 0<v[2]<FAILED_PARAMS
def load_generation(directory, generation=20):
    path=Path(directory)/f"global_gen_{generation}.pkl"
    if not path.is_file(): raise FileNotFoundError(f"Required completed generation checkpoint not found: {path}")
    with path.open("rb") as handle: archive=pickle.load(handle)
    return path, archive, {**archive.get("GLOBAL_DATA_HIST",{}), **archive.get("GLOBAL_DATA",{})}
def select_winner(records):
    eligible=[(g,a["fitness"]) for g,a in records.items() if valid_fitness(a.get("fitness"))]
    if not eligible: raise ValueError("Generation archive contains no valid candidates")
    return max(eligible,key=lambda x:(x[1][1],x[1][0],-x[1][2]))
def file_sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def freeze_manifest(path,value):
    path=Path(path); encoded=json.dumps(value,indent=2,sort_keys=True)+"\n"
    if path.exists():
        if path.read_text()!=encoded: raise ValueError(f"Frozen winner manifest differs from requested selection: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(encoded)
