"""Create one isolated D-FINE variant that reserves half its queries for stride 4."""

import argparse
from pathlib import Path

try:
    from .candidate_runner import FRED_DIR
except ImportError:
    from candidate_runner import FRED_DIR


SEED = FRED_DIR / "seeds" / "seed_dfine.py"
ORIGINAL = """        memory_indices = objectness.topk(min(token_count, self.max_memory_tokens), dim=1).indices
        query_indices = memory_indices[:, :self.num_queries]
"""
REPLACEMENT = """        # Keep the same query and memory limits while testing small-scale selection.
        p4_tokens = features[0].shape[-2] * features[0].shape[-1]
        p4_queries = min(p4_tokens, max(1, self.num_queries // 2))
        p4_indices = objectness[:, :p4_tokens].topk(p4_queries, dim=1).indices
        reserved = torch.zeros_like(objectness, dtype=torch.bool)
        reserved.scatter_(1, p4_indices, True)
        remaining = objectness.masked_fill(reserved, float("-inf")).topk(
            min(token_count, self.max_memory_tokens) - p4_queries, dim=1).indices
        memory_indices = torch.cat((p4_indices, remaining), dim=1)
        query_indices = memory_indices[:, :self.num_queries]
"""


def create_variant(output, seed=SEED):
    source = Path(seed).read_text()
    if source.count(ORIGINAL) != 1:
        raise ValueError("seed query-selection code has changed; review the variant")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(source.replace(ORIGINAL, REPLACEMENT))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create_variant(args.output))


if __name__ == "__main__":
    main()
