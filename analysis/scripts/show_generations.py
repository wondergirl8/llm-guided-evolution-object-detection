"""Print the population of the last N generations of the FRED run, best first.

Usage (from the repo root):
    uv run python analysis/scripts/show_generations.py        # last 2 generations
    uv run python analysis/scripts/show_generations.py 5      # last 5 generations
"""

import os
import pickle
import re
import sys

from deap import base, creator

# Checkpoints pickle DEAP individuals, so these classes must exist to load them.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

CHECKPOINTS = sys.argv[2] if len(sys.argv) > 2 else "fred_yolo11_checkpoints"
FAILED_PARAMS = 999_999_999


def describe(ind):
    map50, map50_95, params = ind.fitness.values
    if params == FAILED_PARAMS or map50 == float("-inf"):
        return f"  {ind[0]}  FAILED (crash or timeout)"
    return f"  {ind[0]}  mAP50 {map50:6.1%}  mAP50:95 {map50_95:6.1%}  params {params:>9,.0f}"


def by_map50(ind):
    return ind.fitness.values[0]


n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
gens = sorted(int(m.group(1)) for f in os.listdir(CHECKPOINTS)
              if (m := re.match(r"checkpoint_gen_(\d+)\.pkl$", f)))
print(f"Latest completed generation: {gens[-1]}")

for gen in gens[-n:]:
    with open(os.path.join(CHECKPOINTS, f"checkpoint_gen_{gen}.pkl"), "rb") as f:
        checkpoint = pickle.load(f)
    print(f"\nGeneration {gen}:")
    for ind in sorted(checkpoint["population"], key=by_map50, reverse=True):
        print(describe(ind))

# The hall of fame in the latest checkpoint holds the best individuals ever seen.
print("\nBest ever (hall of fame, top 3):")
for ind in sorted(checkpoint["hof"], key=by_map50, reverse=True)[:3]:
    print(describe(ind))
