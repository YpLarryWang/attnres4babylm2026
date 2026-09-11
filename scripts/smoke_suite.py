#!/usr/bin/env python3
"""Three short training runs; dynamic intermediate checkpoints plus endpoint fast eval."""

import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--train-python", required=True)
p.add_argument("--eval-python", required=True)
p.add_argument("--eval-repo", required=True, type=Path)
a = p.parse_args()
os.chdir(ROOT)
out = ROOT / "outputs/smoke"
out.mkdir(parents=True, exist_ok=True)


def call(cmd, log):
    with (out / log).open("w") as f:
        subprocess.run(
            list(map(str, cmd)),
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


records = []
for arm in ["baseline", "static", "dynamic"]:
    tag = (
        ""
        if arm == "baseline"
        else "-attnres8" + ("-static" if arm == "static" else "")
    )
    name = f"bl10m-d512L32-do0.1-gate{tag}-offdev-endpoint-b8ga64-s1337-smoke-u6-save2-4-6-eval2x8-nocompile"
    run = ROOT / "outputs/training" / name
    print("TRAIN", arm, flush=True)
    if not (run / "checkpoint_manifest.json").exists():
        call(
            [a.train_python, "scripts/train.py", "--arm", arm, "--smoke"],
            f"{arm}-train.log",
        )
    m = json.loads((run / "checkpoint_manifest.json").read_text())
    assert "final" in m["roles"], m
    ckpts = [run / m["roles"]["final"]]
    if arm == "dynamic":
        ckpts = [run / "ckpt_000002.pt", run / "ckpt_000004.pt", *ckpts]
    for ckpt in ckpts:
        hf = ROOT / "hf-models" / f"smoke-{arm}-{ckpt.stem}"
        if not (hf / "config.json").exists():
            call(
                [
                    a.eval_python,
                    "eval/convert_nanogpt_to_hf.py",
                    "--ckpt",
                    ckpt,
                    "--tokenizer",
                    "tokenizers/10m/tokenizer.json",
                    "--out",
                    hf,
                ],
                f"{hf.name}-export.log",
            )
        call(
            [
                a.eval_python,
                "eval/parity_check.py",
                "--ckpt",
                ckpt,
                "--hf",
                hf,
                "--n",
                "2",
                "--t",
                "32",
                "--tol",
                "0.0001",
            ],
            f"{hf.name}-parity.log",
        )
        result = out / hf.name
        if not (result / "complete.json").exists():
            cmd = [
                a.eval_python,
                "scripts/evaluate.py",
                "--model",
                hf,
                "--eval-repo",
                a.eval_repo,
                "--profile",
                "fast",
                "--batch-size",
                "16",
                "--output-dir",
                result,
            ]
            if ckpt.name != m["roles"]["final"]:
                cmd += ["--only-blimp"]
            else:
                cmd += ["--reading"]
            print("EVAL", hf.name, flush=True)
            call(cmd, f"{hf.name}-eval.log")
        records.append(
            {
                "arm": arm,
                "checkpoint": ckpt.name,
                "model_dir": str(hf),
                "result_dir": str(result),
                "status": "complete",
            }
        )
        (out / "completed-checks.json").write_text(json.dumps(records, indent=2))
(out / "all-complete.json").write_text(
    json.dumps({"complete": True, "evaluated_checkpoints": records}, indent=2)
)
print("ALL SMOKE CHECKS COMPLETE", flush=True)
