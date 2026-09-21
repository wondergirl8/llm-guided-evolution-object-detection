"""
fred_api.py — Loads FRED dataset from HuggingFace using streaming.
No local download required.
"""

from datasets import load_dataset


FRED_REPO = "GabrieleMagrini/FRED"


def get_fred_stream(split: str = "train"):
    """
    Stream FRED dataset directly from HuggingFace.
    split: 'train' or 'test'
    Returns a HuggingFace IterableDataset.
    """
    print(f"Streaming FRED {split} split from HuggingFace...")
    ds = load_dataset(FRED_REPO, streaming=True, split=split)
    return ds


def get_fred_splits():
    """
    Returns both train and test splits as streams.
    """
    train = get_fred_stream("train")
    test = get_fred_stream("test")
    return train, test


if __name__ == "__main__":
    # Quick test — print first sample
    ds = get_fred_stream("train")
    sample = next(iter(ds))
    print("Sample keys:", sample.keys())
    print("Sample:", sample)
