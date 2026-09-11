#!/usr/bin/env python3
"""Download official fast resources at a recorded Hub revision; never alter scorers."""

import argparse, json, zipfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--eval-repo", required=True, type=Path)
p.add_argument("--profile", choices=["fast", "full"], default="fast")
p.add_argument("--dataset-revision", default="8d52da9424a9ff30b9e8266c4f751aba9c504233")
a = p.parse_args()
root = a.eval_repo.resolve() / "strict"
rid = "BabyLM-community/BabyLM-2026-Strict-Evals"
sha = HfApi().repo_info(rid, repo_type="dataset", revision=a.dataset_revision).sha
snapshot_download(
    repo_id=rid,
    repo_type="dataset",
    revision=sha,
    local_dir=root,
    allow_patterns=[f"evaluation_data/{a.profile}_eval/**"],
)
for z in (root / "evaluation_data" / f"{a.profile}_eval").rglob("*.zip"):
    with zipfile.ZipFile(z) as f:
        target = (
            root
            if any(n.startswith("evaluation_data/") for n in f.namelist())
            else z.parent
        )
        for item in f.infolist():
            dest = (target / item.filename).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError("Unsafe ZIP member")
        f.extractall(target, pwd=b"BabyLM2025")
(root / f"{a.profile}-data-provenance.json").write_text(
    json.dumps({"repo": rid, "revision": sha, "profile": a.profile}, indent=2)
)
print("Prepared official evaluation data:", sha)
