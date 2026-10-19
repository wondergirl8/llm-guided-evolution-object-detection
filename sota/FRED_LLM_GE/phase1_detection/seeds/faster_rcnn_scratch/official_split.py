"""Pinned official FRED challenging-test membership resolver."""
import hashlib,re,urllib.request
TEST_SPLIT_URL="https://raw.githubusercontent.com/miccunifi/FRED/2bf89c5376eda528431b62d6c60f2c13d8f95ab4/dataset_splits/challenging/challenging_test_split.txt"
TEST_SPLIT_SHA256="a1f408317eaabbf034ea4c53e40f1cc5fbbc507e58881131ada85768c6afcc56"
# The official challenging split is orthogonal to the Hugging Face repository's
# physical train/test folders. These locations are pinned to fred_stream.REVISION.
HF_TRAIN_MEMBERS={"10","17","22","23","24","27","28","33","35","36","68","69","71","76","77","78","79","81","82","84","85","86","87","88","89","97","98","99","100","105","107","108","109","112","113","140","141","176","177","188","189","190","191","192","193"}
HF_TEST_MEMBERS={"9","21","29","34","83","90","110","111","114","142","178","179","194","195"}
def parse_official_split(text):
    values=[]
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        match=re.fullmatch(r"(\d+)/?",line)
        if not match: raise ValueError(f"Unexpected official split entry: {line!r}")
        values.append(match.group(1))
    if not values or len(values)!=len(set(values)): raise ValueError("Official split empty or duplicated")
    return values
def official_test_sequences():
    payload=urllib.request.urlopen(TEST_SPLIT_URL,timeout=30).read()
    if hashlib.sha256(payload).hexdigest()!=TEST_SPLIT_SHA256: raise ValueError("Official challenging_test split checksum mismatch")
    return parse_official_split(payload.decode())
def official_test_archives():
    sequences=official_test_sequences(); members=set(sequences)
    if HF_TRAIN_MEMBERS&HF_TEST_MEMBERS or HF_TRAIN_MEMBERS|HF_TEST_MEMBERS!=members:
        raise ValueError("Pinned Hugging Face archive mapping does not match official challenging_test")
    return [("train" if sequence in HF_TRAIN_MEMBERS else "test",sequence) for sequence in sequences]
