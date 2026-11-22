"""Generation-20 archive loading and deterministic winner selection."""
import hashlib, json, pickle, math, os, tempfile
from pathlib import Path
FAILED_PARAMS = 999_999_999
def valid_fitness(v):
    try:
        return (isinstance(v,(tuple,list)) and len(v)==3 and all(math.isfinite(x) for x in v)
                and 0<=v[1]<=v[0]<=1 and 0<v[2]<FAILED_PARAMS and float(v[2]).is_integer())
    except (TypeError, ValueError):
        return False

def pareto_front(records):
    eligible = {g: tuple(a["fitness"]) for g, a in records.items() if valid_fitness(a.get("fitness"))}
    if not eligible:
        raise ValueError("Generation archive contains no valid candidates")
    return [(g, score) for g, score in sorted(eligible.items()) if not any(
        other[0] >= score[0] and other[1] >= score[1] and other[2] <= score[2]
        and other != score for other in eligible.values())]
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
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(encoded)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != encoded:
                raise ValueError(f"Frozen manifest differs: {path}")
    finally:
        os.unlink(temporary)
