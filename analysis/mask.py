#!/usr/bin/env python3
"""Reproduce trained dynamic-model routing masks with the original intervention code."""

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True, type=Path)
p.add_argument("--eval-repo", required=True, type=Path)
p.add_argument("--mode", choices=["none", "nonrecent", "random_count_matched"], required=True)
p.add_argument("--mask-seed", type=int, default=20260718)
p.add_argument("--output-dir", required=True, type=Path)
a = p.parse_args()
output = a.output_dir.resolve()
hf = output / "model"
output.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable,
        str(ROOT / "eval/convert_nanogpt_to_hf.py"),
        "--ckpt",
        str(a.checkpoint.resolve()),
        "--tokenizer",
        str(ROOT / "tokenizers/10m/tokenizer.json"),
        "--out",
        str(hf),
    ],
    check=True,
)
config = json.loads((hf / "config.json").read_text())
assert config["use_attn_res"] and not config.get("use_static_attn_res", False), (
    "The masking analysis is defined only for dynamic models"
)
shutil.copy2(ROOT / "analysis/masking/modeling_nanogpt.py", hf / "modeling_nanogpt.py")
# Verify the instrumented wrapper with masks off before applying any intervention.
env = dict(os.environ, NANOGPT_ATTNRES_MASK="none")
subprocess.run(
    [
        sys.executable,
        str(ROOT / "eval/parity_check.py"),
        "--ckpt",
        str(a.checkpoint.resolve()),
        "--hf",
        str(hf),
        "--n",
        "2",
        "--t",
        "32",
        "--tol",
        "0.0001",
    ],
    env=env,
    check=True,
)
env.update(NANOGPT_ATTNRES_MASK=a.mode, NANOGPT_ATTNRES_MASK_SEED=str(a.mask_seed))
subprocess.run(
    [
        sys.executable,
        str(ROOT / "scripts/evaluate.py"),
        "--model",
        str(hf),
        "--eval-repo",
        str(a.eval_repo.resolve()),
        "--profile",
        "full",
        "--only-blimp",
        "--output-dir",
        str(output / "results"),
    ],
    env=env,
    check=True,
)
(output / "mask-provenance.json").write_text(
    json.dumps(
        {"mode": a.mode, "mask_seed": a.mask_seed, "final_router_masked": False},
        indent=2,
    )
)
