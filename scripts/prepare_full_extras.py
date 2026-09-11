#!/usr/bin/env python3
"""Run official full-suite downloaders with the paper's dataset revisions pinned."""

import argparse
import json
import os
from pathlib import Path
import runpy

import datasets
import nltk

REVISIONS = {
    "mrlbenchmarks/global-piqa-parallel": "b0b18516a8bc2cb1106bce3dd4db32848ca715ea",
    "mrlbenchmarks/global-piqa-nonparallel": "6777742fa3634c0583cda3b7f8a482ea7b1b0937",
    "ewok-core/ewok-core-1.0": "34d912a608066c92e2990a0328ffc3bd9a716042",
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--eval-repo", required=True, type=Path)
a = p.parse_args()
strict = a.eval_repo.resolve() / "strict"
os.chdir(strict)
(strict / "evaluation_data/full_eval").mkdir(parents=True, exist_ok=True)
if not nltk.download("punkt_tab", quiet=True):
    raise RuntimeError("Could not install NLTK tokenization data")
original = datasets.load_dataset


def pinned_load_dataset(path, *args, **kwargs):
    if path not in REVISIONS:
        raise ValueError(f"Unexpected dataset in official downloader: {path}")
    kwargs["revision"] = REVISIONS[path]
    return original(path, *args, **kwargs)


# Only dataset resolution changes. Official filtering and serialization run intact.
datasets.load_dataset = pinned_load_dataset
try:
    runpy.run_path("evaluation_pipeline/global_piqa/dl.py", run_name="__main__")
    runpy.run_path("evaluation_pipeline/ewok/dl_and_filter.py", run_name="__main__")
finally:
    datasets.load_dataset = original
(strict / "full-extra-data-provenance.json").write_text(json.dumps(REVISIONS, indent=2))
print("Prepared pinned official Global PIQA and EWoK datasets")
