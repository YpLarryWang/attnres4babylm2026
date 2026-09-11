#!/usr/bin/env python3
"""Aggregate the 21 paper endpoints; six-suite means exclude Reading and AoA."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, default=ROOT / "results/paper/offdev_results.csv")
p.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/endpoints.json")
a = p.parse_args()
groups = defaultdict(list)
with a.input.open() as f:
    for row in csv.DictReader(f):
        name = row["run_name"]
        arm = (
            "static"
            if "-static-" in name
            else "dynamic"
            if "-attnres" in name
            else "baseline"
        )
        scores = {
            key: float(row[key])
            for key in [
                "blimp",
                "supplement",
                "ewok",
                "entity_tracking",
                "comps",
                "full_dev_nll",
                "full_test_nll",
            ]
        }
        scores["global_piqa"] = mean(
            float(row[key])
            for key in ["global_piqa_parallel", "global_piqa_nonparallel"]
        )
        scores["zero_shot_mean6"] = mean(
            scores[key]
            for key in [
                "blimp",
                "supplement",
                "ewok",
                "entity_tracking",
                "comps",
                "global_piqa",
            ]
        )
        groups[(row["train_words"], int(row["n_layer"]), arm)].append(
            (int(row["seed"]), scores)
        )
assert len(groups) == 7
output = []
for (track, depth, arm), entries in sorted(groups.items()):
    assert sorted(seed for seed, _ in entries) == [1337, 1338, 1339]
    summary = {}
    for metric in entries[0][1]:
        values = [scores[metric] for _, scores in entries]
        summary[metric] = {"mean": mean(values), "sample_sd": stdev(values), "n": 3}
    output.append({"track": track, "depth": depth, "arm": arm, "metrics": summary})
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(output, indent=2) + "\n")
print(f"Aggregated 21 endpoints into seven three-seed groups: {a.output}")
