#!/usr/bin/env python3
"""Recompute paper statistics from portable, frozen evaluation evidence."""

import argparse, csv, json, math
from pathlib import Path
import numpy as np
from diag_supp_analysis import (
    read_csv,
    trajectory_summaries,
    masking_tables,
    fresh_100m_summary,
)

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input-dir", type=Path, default=ROOT / "results/paper")
p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/analysis")
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)
b = read_csv(a.input_dir / "diag_supp_behavior_long.csv")
d = read_csv(a.input_dir / "diag_supp_dev_loss.csv")
trajectory, _ = trajectory_summaries(b, d)
draws, contrasts, correlations = masking_tables(b)
fresh, _ = fresh_100m_summary(b, d)
for name, rows in [
    ("trajectory", trajectory),
    ("masking_draws", draws),
    ("masking_contrasts", contrasts),
    ("masking_correlations", correlations),
    ("fresh_100m", fresh),
]:
    with (a.output_dir / f"{name}.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
# Compare computed output with frozen published-analysis tables, not implementation-only tests.
for name, rows, source in [
    ("trajectory", trajectory, "diag_supp_trajectory_summary.csv"),
    ("fresh_100m", fresh, "diag_supp_100m_fresh_summary.csv"),
    ("masking_draws", draws, "diag_supp_masking_draws.csv"),
    ("masking_contrasts", contrasts, "diag_supp_masking_contrasts.csv"),
    ("masking_correlations", correlations, "diag_supp_masking_correlations.csv"),
]:
    expected = read_csv(a.input_dir / source)
    assert len(rows) == len(expected), (name, len(rows), len(expected))
    for got, want in zip(rows, expected):
        for key, value in got.items():
            try:
                assert math.isclose(
                    float(value), float(want[key]), abs_tol=1e-9, rel_tol=1e-9
                ), (name, key, value, want[key])
            except (ValueError, TypeError):
                assert str(value) == want[key], (name, key, value, want[key])
print(
    json.dumps(
        {
            "paper_analysis_verified": True,
            "trajectory_rows": len(trajectory),
            "mask_draw_rows": len(draws),
            "output_dir": str(a.output_dir),
        }
    )
)
