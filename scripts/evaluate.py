#!/usr/bin/env python3
"""Fail-fast entry point for an explicitly versioned official BabyLM evaluator."""

import argparse, json, subprocess, sys, os
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--eval-repo", required=True, type=Path)
p.add_argument("--profile", choices=["fast", "full"], default="fast")
p.add_argument("--revision", default="main")
p.add_argument("--only-blimp", action="store_true")
p.add_argument("--reading", action="store_true")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
a = p.parse_args()
root = a.eval_repo.resolve()
strict = root / "strict"
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
model = str(Path(a.model).resolve()) if Path(a.model).exists() else a.model
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
(out / "evaluation-provenance.json").write_text(
    json.dumps(
        {
            "official_code_commit": sha,
            "profile": a.profile,
            "model": model,
            "revision": a.revision,
            "batch_size": a.batch_size,
        },
        indent=2,
    )
)
if a.profile == "fast":
    pairs = [
        ("blimp", "blimp_fast"),
        ("blimp", "supplement_fast"),
        ("ewok", "ewok_fast"),
        ("entity_tracking", "entity_tracking_fast"),
    ]
else:
    pairs = [
        ("blimp", "blimp_filtered"),
        ("blimp", "supplement_filtered"),
        ("ewok", "ewok_filtered"),
        ("entity_tracking", "entity_tracking"),
        ("comps", "comps"),
        ("global_piqa_parallel", "global_piqa_parallel"),
        ("global_piqa_nonparallel", "global_piqa_nonparallel"),
    ]
if a.only_blimp:
    pairs = pairs[:1]
for task, data in pairs:
    path = strict / "evaluation_data" / f"{a.profile}_eval" / data
    if not path.is_dir() or not any(path.rglob("*.json*")):
        raise SystemExit("Missing evaluation data: " + str(path))
    cmd = [
        sys.executable,
        "-m",
        "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name",
        model,
        "--backend",
        "causal",
        "--task",
        task,
        "--data_path",
        str(path),
        "--revision_name",
        a.revision,
        "--batch_size",
        str(a.batch_size),
        "--save_predictions",
        "--output_dir",
        str(out),
    ]
    subprocess.run(cmd, cwd=strict, check=True)
    result = out / Path(model).stem / a.revision / "zero_shot/causal" / task / data
    report = result / "best_temperature_report.txt"
    predictions = result / "predictions.json"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("Missing report: " + str(report))
    if not predictions.is_file() or not json.loads(predictions.read_text()):
        raise RuntimeError("Missing or empty predictions: " + str(predictions))
if a.reading:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation_pipeline.reading.run",
            "--model_path_or_name",
            model,
            "--backend",
            "causal",
            "--data_path",
            str(
                strict
                / "evaluation_data"
                / f"{a.profile}_eval"
                / "reading/reading_data.csv"
            ),
            "--revision_name",
            a.revision,
            "--output_dir",
            str(out),
        ],
        cwd=strict,
        check=True,
    )
    if not (
        out
        / Path(model).stem
        / a.revision
        / "zero_shot/causal/reading/predictions.json"
    ).is_file():
        raise RuntimeError("Missing Reading predictions")
(out / "complete.json").write_text(
    json.dumps(
        {
            "complete": True,
            "tasks": [d for _, d in pairs],
            "reading": a.reading,
            "official_commit": sha,
        },
        indent=2,
    )
)
