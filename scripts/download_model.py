#!/usr/bin/env python3
"""Download an indexed paper endpoint at its verified immutable Hub revision."""

import argparse, json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--track", choices=["10m", "100m"], required=True)
p.add_argument("--depth", type=int, choices=[16, 32], default=32)
p.add_argument("--arm", choices=["baseline", "static", "dynamic"], required=True)
p.add_argument("--seed", type=int, choices=[1337, 1338, 1339], default=1337)
p.add_argument("--output-dir", type=Path)
a = p.parse_args()
models = json.loads((ROOT / "docs/models.json").read_text())
matched = [
    m
    for m in models
    if m["track"].lower() == a.track
    and m["layers"] == a.depth
    and m["arm"] == a.arm
    and m["seed"] == a.seed
]
if len(matched) != 1:
    p.error("No trained endpoint exists for this configuration.")
m = matched[0]
out = a.output_dir or ROOT / "hf-models" / m["repo_id"].split("/")[-1]
snapshot_download(repo_id=m["repo_id"], revision=m["revision"], local_dir=out)
print(out.resolve())
