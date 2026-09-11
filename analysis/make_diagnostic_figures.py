#!/usr/bin/env python3
"""Render reproducible, paper-ready AttnRes diagnostic figures.

The script intentionally writes every panel as a separate PNG and SVG so that
LaTeX controls panel composition.  It also exports chart-data CSVs for the two
analyses whose values are not already present in a source table.

Example:
    python scripts/make_diagnostic_figures.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


SEEDS = (1337, 1338, 1339)
POSITION_BINS = ((1, 4), (5, 16), (17, 64), (65, 256), (257, 512))
LATE_100M_LABELS = ("50M", "60M", "70M", "80M", "90M")
MATCHED_100M_LABELS = ("10M", "20M", "30M", "40M", "50M", "60M", "70M")
COLORS = {
    "teal": "#087E8B",
    "orange": "#D97706",
    "gray": "#64748B",
    "light_gray": "#CBD5E1",
    "dark": "#1F2937",
    "grid": "#E5E7EB",
}
METRICS = {
    "overall_blimp": ("blimp", "overall", "overall"),
    "filler_gap_dependency": ("blimp", "linguistics_term", "filler_gap_dependency"),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows: list[dict], fields: tuple[str, ...], label: str) -> dict:
    index = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in index:
            raise ValueError(f"duplicate {label} key: {key}")
        index[key] = row
    return index


def pava_nonincreasing(values) -> np.ndarray:
    """Equal-weight PAVA projection onto a non-increasing sequence."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("PAVA requires a finite one-dimensional sequence")
    blocks = []
    for index, value in enumerate(values):
        blocks.append([index, index + 1, 1.0, float(value)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] >= right[3] / right[2]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    fitted = np.empty_like(values)
    for start, end, weight, total in blocks:
        fitted[start:end] = total / weight
    return fitted


def interpolate_on_nonincreasing_x(target: float, x, y) -> float | None:
    """Linearly interpolate y at target x, without extrapolation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) == 0 or np.any(np.diff(x) > 1e-12):
        raise ValueError("x must be a non-increasing sequence matching y")
    if target > x[0] + 1e-12 or target < x[-1] - 1e-12:
        return None
    exact = np.isclose(x, target, rtol=0.0, atol=1e-12)
    if exact.any():
        return float(y[exact].mean())
    for index in range(len(x) - 1):
        high, low = x[index], x[index + 1]
        if high == low:
            continue
        if high > target > low:
            fraction = (high - target) / (high - low)
            return float(y[index] + fraction * (y[index + 1] - y[index]))
    raise RuntimeError(f"failed to interpolate in-range target {target}")


def behavior_index(rows: list[dict]) -> dict:
    return unique_index(
        rows,
        (
            "corpus", "seed", "architecture", "checkpoint_label", "mask_mode",
            "mask_seed", "task", "level", "term",
        ),
        "behavior",
    )


def dev_index(rows: list[dict]) -> dict:
    return unique_index(
        rows,
        ("corpus", "seed", "architecture", "checkpoint_label"),
        "dev",
    )


def checkpoint_sort_key(label: str) -> tuple[int, int]:
    return (1, 0) if label == "final" else (0, int(label.removesuffix("M")))


def metric_value(index: dict, corpus: str, seed: int, architecture: str,
                 label: str, spec: tuple[str, str, str]) -> float:
    task, level, term = spec
    key = (corpus, str(seed), architecture, label, "none", "", task, level, term)
    return float(index[key]["accuracy"])


def paired_series(behavior: list[dict], dev: list[dict], corpus: str,
                  seed: int, spec: tuple[str, str, str]) -> dict:
    bidx = behavior_index(behavior)
    didx = dev_index(dev)
    labels = sorted(
        {
            key[3] for key in didx
            if key[0] == corpus and key[1] == str(seed) and key[2] == "baseline"
            and (key[3] != "final" or corpus == "10m")
        },
        key=checkpoint_sort_key,
    )
    expected = 19 if corpus == "10m" else 9
    if len(labels) != expected:
        raise RuntimeError(f"{corpus} seed {seed}: expected {expected} checkpoints, got {labels}")
    base_dev = [didx[(corpus, str(seed), "baseline", label)] for label in labels]
    attn_dev = [didx[(corpus, str(seed), "attnres", label)] for label in labels]
    words = np.asarray([float(row["words_seen"]) for row in base_dev])
    attn_words = np.asarray([float(row["words_seen"]) for row in attn_dev])
    if not np.array_equal(words, attn_words):
        raise RuntimeError(f"paired words differ for {corpus} seed {seed}")
    return {
        "labels": labels,
        "words": words,
        "iters": np.asarray([int(row["iter_num"]) for row in base_dev]),
        "baseline_accuracy": np.asarray([
            metric_value(bidx, corpus, seed, "baseline", label, spec) for label in labels
        ]),
        "attnres_accuracy": np.asarray([
            metric_value(bidx, corpus, seed, "attnres", label, spec) for label in labels
        ]),
        "baseline_nll": np.asarray([float(row["mean_nll"]) for row in base_dev]),
        "attnres_nll": np.asarray([float(row["mean_nll"]) for row in attn_dev]),
    }


def matched_delta(series: dict) -> dict:
    """Compare AttnRes accuracy with baseline accuracy interpolated at equal NLL."""
    baseline_nll = pava_nonincreasing(series["baseline_nll"])
    result = {}
    for label, words, target, accuracy in zip(
        series["labels"], series["words"], series["attnres_nll"], series["attnres_accuracy"]
    ):
        comparator = interpolate_on_nonincreasing_x(float(target), baseline_nll,
                                                    series["baseline_accuracy"])
        if comparator is not None:
            result[label] = {
                "words": float(words),
                "delta": float(accuracy - comparator),
                "attnres_accuracy": float(accuracy),
                "interpolated_baseline_accuracy": comparator,
                "attnres_nll": float(target),
            }
    return result


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": COLORS["dark"],
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
    })


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(axis, xlog: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=COLORS["grid"], lw=0.7, zorder=0)
    if xlog:
        axis.set_xscale("log")
        ticks = np.asarray([1, 2, 5, 10, 20, 30, 50, 90]) * 1e6
        axis.set_xticks(ticks, ["1", "2", "5", "10", "20", "30", "50", "90"])
        axis.minorticks_off()
        axis.set_xlabel("Words seen (M)")


def render_schematic(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.65))
    panels = (
        (axes[0], "Standard residual", False),
        (axes[1], "Block AttnRes", True),
    )
    for axis, title, routed in panels:
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 4.5)
        axis.axis("off")
        axis.set_title(title, loc="left", fontweight="semibold")
        boxes = [(0.4, 1.65, "Embedding"), (3.7, 1.65, "Block 1"), (7.0, 1.65, "Block 2")]
        for x, y, label in boxes:
            box = FancyBboxPatch((x, y), 2.2, 1.0, boxstyle="round,pad=0.08",
                                 fc="white", ec=COLORS["dark"], lw=1.0)
            axis.add_patch(box)
            axis.text(x + 1.1, y + 0.5, label, ha="center", va="center")
        axis.add_patch(FancyArrowPatch((2.6, 2.15), (3.7, 2.15), arrowstyle="->",
                                       mutation_scale=10, color=COLORS["gray"]))
        axis.add_patch(FancyArrowPatch((5.9, 2.15), (7.0, 2.15), arrowstyle="->",
                                       mutation_scale=10, color=COLORS["gray"]))
        if routed:
            axis.add_patch(FancyArrowPatch((1.5, 2.72), (7.35, 2.68), arrowstyle="->",
                                           connectionstyle="arc3,rad=-0.24", mutation_scale=10,
                                           color=COLORS["teal"], lw=1.5))
            axis.add_patch(FancyArrowPatch((4.8, 2.72), (8.0, 2.68), arrowstyle="->",
                                           connectionstyle="arc3,rad=-0.22", mutation_scale=10,
                                           color=COLORS["teal"], lw=1.5))
            axis.text(6.0, 3.65, "learned depth weights", ha="center", color=COLORS["teal"])
            axis.text(8.1, 0.85, r"$h_2=\sum_i\alpha_{i\to2}v_i$", ha="center")
        else:
            axis.add_patch(FancyArrowPatch((1.5, 1.58), (8.1, 1.58), arrowstyle="->",
                                           connectionstyle="arc3,rad=0.25", mutation_scale=10,
                                           color=COLORS["gray"], lw=1.2))
            axis.text(8.1, 0.85, r"$h_2=h_0+v_0+v_1$", ha="center")
    fig.suptitle("Residual accumulation versus learned routing across depth", y=1.02,
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_method_attnres_schematic")


def render_trajectory_delta(series_by_seed: dict, field: str, ylabel: str, title: str,
                            subtitle: str, stem: str, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 3.65))
    values = []
    words = []
    for seed in SEEDS:
        series = series_by_seed[seed]
        if field == "dev_nll":
            delta = series["baseline_nll"] - series["attnres_nll"]
        else:
            delta = series["attnres_accuracy"] - series["baseline_accuracy"]
        values.append(delta)
        words.append(series["words"])
        axis.plot(series["words"], delta, color=COLORS["teal"], alpha=0.32,
                  lw=1.0, marker="o", ms=2.7, zorder=2)
        prewarm = series["iters"] < 40
        axis.scatter(series["words"][prewarm], delta[prewarm], s=24, facecolors="white",
                     edgecolors=COLORS["teal"], alpha=0.6, lw=0.9, zorder=3)
    mean_words = np.mean(words, axis=0)
    values = np.asarray(values)
    mean = values.mean(axis=0)
    axis.fill_between(mean_words, values.min(axis=0), values.max(axis=0),
                      color=COLORS["teal"], alpha=0.10, lw=0, zorder=1)
    axis.plot(mean_words, mean, color=COLORS["teal"], lw=2.4,
              label="mean of 3 paired seeds", zorder=4)
    axis.axhline(0, color=COLORS["gray"], lw=0.9)
    axis.axvline(8e6, color=COLORS["gray"], lw=0.9, ls="--")
    axis.text(8e6, axis.get_ylim()[1], " warmup end", ha="left", va="top",
              fontsize=7.5, color=COLORS["gray"])
    style_axis(axis, xlog=True)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="semibold", pad=18)
    axis.text(0, 1.02, subtitle, transform=axis.transAxes, fontsize=8,
              color=COLORS["gray"], va="bottom")
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def render_matched_blimp(series_by_seed: dict, output_dir: Path) -> None:
    matched = {seed: matched_delta(series_by_seed[seed]) for seed in SEEDS}
    common = [label for label in series_by_seed[SEEDS[0]]["labels"]
              if all(label in matched[seed] for seed in SEEDS)]
    fig, axis = plt.subplots(figsize=(5.2, 3.65))
    for seed in SEEDS:
        labels = [label for label in series_by_seed[seed]["labels"] if label in matched[seed]]
        x = [matched[seed][label]["words"] for label in labels]
        y = [matched[seed][label]["delta"] for label in labels]
        axis.plot(x, y, color=COLORS["teal"], alpha=0.32, lw=1.0,
                  marker="o", ms=2.7, zorder=2)
    common_x = np.asarray([
        np.mean([matched[seed][label]["words"] for seed in SEEDS]) for label in common
    ])
    common_y = np.asarray([
        [matched[seed][label]["delta"] for label in common] for seed in SEEDS
    ]).T
    axis.fill_between(common_x, common_y.min(axis=1), common_y.max(axis=1),
                      color=COLORS["teal"], alpha=0.10, lw=0)
    axis.plot(common_x, common_y.mean(axis=1), color=COLORS["teal"], lw=2.4,
              label="mean where all seeds overlap", zorder=4)
    axis.axhline(0, color=COLORS["gray"], lw=0.9)
    axis.axvline(8e6, color=COLORS["gray"], lw=0.9, ls="--")
    style_axis(axis, xlog=True)
    axis.set_ylabel("AttnRes − interpolated baseline BLiMP (pp)")
    axis.set_title("Overall BLiMP at matched dev NLL", loc="left",
                   fontweight="semibold", pad=18)
    axis.text(0, 1.02, "Within-seed interpolation; no extrapolation; n=3 paired seeds",
              transform=axis.transAxes, fontsize=8, color=COLORS["gray"], va="bottom")
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_dev_blimp_matched_10m")

    rows = []
    for seed in SEEDS:
        all_labels = series_by_seed[seed]["labels"]
        for label in all_labels:
            if label in matched[seed]:
                rows.append({"seed": seed, "checkpoint_label": label, **matched[seed][label],
                             "all_seed_overlap": int(label in common)})
            else:
                rows.append({
                    "seed": seed, "checkpoint_label": label,
                    "words": float(series_by_seed[seed]["words"][all_labels.index(label)]),
                    "delta": "", "attnres_accuracy": "",
                    "interpolated_baseline_accuracy": "",
                    "attnres_nll": float(series_by_seed[seed]["attnres_nll"][all_labels.index(label)]),
                    "all_seed_overlap": 0,
                })
    write_csv(output_dir / "fig_dev_blimp_matched_10m.csv", rows,
              ("seed", "checkpoint_label", "words", "delta", "attnres_accuracy",
               "interpolated_baseline_accuracy", "attnres_nll", "all_seed_overlap"))


def weighted_position_bins(contexts, deltas, counts) -> list[dict]:
    contexts = np.asarray(contexts, dtype=np.int64)
    deltas = np.asarray(deltas, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    output = []
    for start, end in POSITION_BINS:
        keep = (contexts >= start) & (contexts <= end)
        weight = int(counts[keep].sum())
        output.append({
            "start": start,
            "end": end,
            "x": math.sqrt(start * end),
            "delta": float(np.dot(counts[keep], deltas[keep]) / weight),
        })
    return output


def render_position_nll(position: list[dict], dev: list[dict], output_dir: Path) -> None:
    pidx = unique_index(position, ("corpus", "seed", "architecture", "checkpoint_label", "loss_index"),
                        "position")
    didx = dev_index(dev)
    curves = {}
    for seed in SEEDS:
        contexts, deltas, counts = [], [], []
        for loss_index in range(512):
            baseline = pidx[("10m", str(seed), "baseline", "final", str(loss_index))]
            attnres = pidx[("10m", str(seed), "attnres", "final", str(loss_index))]
            if int(baseline["token_count"]) != int(attnres["token_count"]):
                raise RuntimeError(f"position counts differ for seed {seed}, index {loss_index}")
            contexts.append(int(baseline["context_length"]))
            counts.append(int(baseline["token_count"]))
            deltas.append(float(baseline["mean_nll"]) - float(attnres["mean_nll"]))
        overall = (float(didx[("10m", str(seed), "baseline", "final")]["mean_nll"])
                   - float(didx[("10m", str(seed), "attnres", "final")]["mean_nll"]))
        weighted = float(np.dot(counts, deltas) / np.sum(counts))
        if not math.isclose(overall, weighted, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"position/headline mismatch for seed {seed}")
        curves[seed] = {"bins": weighted_position_bins(contexts, deltas, counts),
                        "overall": overall}
    fig, axis = plt.subplots(figsize=(5.2, 3.65))
    seed_values = []
    for seed in SEEDS:
        x = [row["x"] for row in curves[seed]["bins"]]
        y = [row["delta"] for row in curves[seed]["bins"]]
        seed_values.append(y)
        axis.plot(x, y, color=COLORS["teal"], alpha=0.32, lw=1.0, marker="o", ms=3)
    axis.plot(x, np.mean(seed_values, axis=0), color=COLORS["teal"], lw=2.4,
              marker="o", label="mean of 3 paired seeds")
    overall = float(np.mean([curves[seed]["overall"] for seed in SEEDS]))
    axis.axhline(overall, color=COLORS["gray"], lw=1.0, ls="--",
                 label=f"token-weighted overall: {overall:.3f}")
    axis.axhline(0, color=COLORS["gray"], lw=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [f"{a}–{b}" for a, b in POSITION_BINS])
    axis.minorticks_off()
    axis.set_xlabel("Context-position bin")
    axis.set_ylabel("Baseline − AttnRes dev NLL (nats)")
    axis.set_title("The final NLL gain is not concentrated at long context",
                   loc="left", fontweight="semibold", pad=18)
    axis.text(0, 1.02, "Token-weighted position bins; final 10M checkpoints",
              transform=axis.transAxes, fontsize=8, color=COLORS["gray"], va="bottom")
    style_axis(axis)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_position_nll_final_10m")


def routing_rows(contrasts: list[dict]) -> list[dict]:
    rows = [row for row in contrasts if row["training_seed"] == "mean"]
    if len(rows) != 13:
        raise RuntimeError(f"expected 13 mean routing rows, got {len(rows)}")
    return rows


def render_routing(contrasts: list[dict], x_field: str, xlabel: str, title: str,
                   subtitle: str, stem: str, output_dir: Path) -> None:
    rows = routing_rows(contrasts)
    x = np.asarray([float(row[x_field]) for row in rows])
    y = np.asarray([float(row["nonrecent_excess_cost_pp"]) for row in rows])
    low = np.asarray([float(row["draw_aggregate_min_pp"]) for row in rows])
    high = np.asarray([float(row["draw_aggregate_max_pp"]) for row in rows])
    fig, axis = plt.subplots(figsize=(5.2, 4.0))
    axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="o",
                  color=COLORS["teal"], ecolor=COLORS["light_gray"],
                  capsize=2.5, ms=5, zorder=3)
    axis.axhline(0, color=COLORS["gray"], lw=0.8)
    axis.axvline(0, color=COLORS["gray"], lw=0.8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"Nonrecent masking cost $\bar{A}_R-A_N$ (pp)")
    axis.set_title(title, loc="left", fontweight="semibold", pad=18)
    axis.text(0, 1.02, subtitle, transform=axis.transAxes, fontsize=8,
              color=COLORS["gray"], va="bottom")
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def term_specs(behavior: list[dict]) -> dict[str, tuple[str, str, str]]:
    terms = sorted({
        row["term"] for row in behavior
        if row["corpus"] == "100m" and row["mask_mode"] == "none"
        and row["task"] == "blimp" and row["level"] == "linguistics_term"
    })
    if len(terms) != 13:
        raise RuntimeError(f"expected 13 BLiMP terms, got {terms}")
    return {term: ("blimp", "linguistics_term", term) for term in terms}


def forest_rows(behavior: list[dict], dev: list[dict]) -> list[dict]:
    output = []
    for term, spec in term_specs(behavior).items():
        seed_means = []
        matched_labels = []
        for seed in SEEDS:
            series = paired_series(behavior, dev, "100m", seed, spec)
            matched = matched_delta(series)
            labels = [label for label in LATE_100M_LABELS if label in matched]
            if labels != ["50M", "60M", "70M"]:
                raise RuntimeError(f"unexpected 100M overlap for {term}, seed {seed}: {labels}")
            seed_means.append(float(np.mean([matched[label]["delta"] for label in labels])))
            matched_labels.append("/".join(labels))
        output.append({
            "term": term,
            "mean_delta_pp": float(np.mean(seed_means)),
            "min_seed_delta_pp": float(np.min(seed_means)),
            "max_seed_delta_pp": float(np.max(seed_means)),
            "seed_1337_delta_pp": seed_means[0],
            "seed_1338_delta_pp": seed_means[1],
            "seed_1339_delta_pp": seed_means[2],
            "positive_seed_pairs": int(np.sum(np.asarray(seed_means) > 0)),
            "matched_labels_per_seed": ";".join(matched_labels),
        })
    return sorted(output, key=lambda row: float(row["mean_delta_pp"]))


def render_forest(behavior: list[dict], dev: list[dict], output_dir: Path) -> None:
    rows = forest_rows(behavior, dev)
    y = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(6.6, 5.4))
    for index, row in enumerate(rows):
        color = COLORS["orange"] if row["term"] == "filler_gap_dependency" else COLORS["teal"]
        seed_values = [float(row[f"seed_{seed}_delta_pp"]) for seed in SEEDS]
        axis.hlines(index, min(seed_values), max(seed_values), color=COLORS["light_gray"], lw=2)
        axis.scatter(seed_values, [index] * 3, color=COLORS["gray"], s=18, alpha=0.75, zorder=2)
        axis.scatter(float(row["mean_delta_pp"]), index, color=color, marker="D", s=38, zorder=3)
    axis.axvline(0, color=COLORS["gray"], lw=0.9)
    axis.set_yticks(y, [row["term"].replace("_", " ") for row in rows])
    axis.set_xlabel("AttnRes − interpolated baseline accuracy (pp)")
    axis.set_title("100M term-level behavior at matched dev NLL", loc="left",
                   fontweight="semibold", pad=18)
    axis.text(0, 1.02,
              "Mean over 50/60/70M matched checkpoints; diamonds are seed means; lines are seed ranges",
              transform=axis.transAxes, fontsize=8, color=COLORS["gray"], va="bottom")
    style_axis(axis)
    axis.grid(axis="x", color=COLORS["grid"], lw=0.7)
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_100m_term_matched_forest")
    write_csv(
        output_dir / "fig_100m_term_matched_forest.csv", rows,
        ("term", "mean_delta_pp", "min_seed_delta_pp", "max_seed_delta_pp",
         "seed_1337_delta_pp", "seed_1338_delta_pp", "seed_1339_delta_pp",
         "positive_seed_pairs", "matched_labels_per_seed"),
    )


def render_term_heatmap(behavior: list[dict], dev: list[dict], output_dir: Path) -> None:
    """Show the full matched-NLL category-by-checkpoint matrix for 100M runs."""
    terms = list(term_specs(behavior))
    matrix = np.empty((len(terms), len(MATCHED_100M_LABELS)), dtype=np.float64)
    csv_rows = []
    for term_index, (term, spec) in enumerate(term_specs(behavior).items()):
        seed_matches = {
            seed: matched_delta(paired_series(behavior, dev, "100m", seed, spec))
            for seed in SEEDS
        }
        row = {"term": term}
        for label_index, label in enumerate(MATCHED_100M_LABELS):
            missing = [seed for seed in SEEDS if label not in seed_matches[seed]]
            if missing:
                raise RuntimeError(
                    f"100m {term} {label}: no matched-NLL value for seeds {missing}"
                )
            value = float(np.mean([
                seed_matches[seed][label]["delta"] for seed in SEEDS
            ]))
            matrix[term_index, label_index] = value
            row[label] = value
        row["mean_10M_70M"] = float(matrix[term_index].mean())
        csv_rows.append(row)

    display = np.column_stack([matrix, matrix.mean(axis=1)])
    extent = float(np.max(np.abs(display)))
    cmap = LinearSegmentedColormap.from_list(
        "attnres_diverging", ("#C66A1B", "#F8FAFC", "#087E8B")
    )
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    fig, axis = plt.subplots(figsize=(8.0, 5.7))
    image = axis.imshow(display, cmap=cmap, norm=norm, aspect="auto")
    xlabels = [*MATCHED_100M_LABELS, "Mean"]
    axis.set_xticks(np.arange(len(xlabels)), xlabels)
    axis.set_yticks(
        np.arange(len(terms)),
        [term.replace("_", " ") for term in terms],
    )
    axis.set_xlabel("AttnRes checkpoint (words seen)")
    axis.set_title(
        "100M BLiMP category residuals at matched dev NLL",
        loc="left", fontweight="semibold", pad=24,
    )
    axis.text(
        0, 1.025,
        "Cells are AttnRes minus interpolated baseline accuracy, averaged over three paired seeds (pp)",
        transform=axis.transAxes, fontsize=8, color=COLORS["gray"], va="bottom",
    )
    axis.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(terms), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.axvline(len(MATCHED_100M_LABELS) - 0.5, color=COLORS["dark"], lw=1.5)
    for row_index in range(display.shape[0]):
        for column_index in range(display.shape[1]):
            value = display[row_index, column_index]
            text_color = "white" if abs(value) > 0.53 * extent else COLORS["dark"]
            axis.text(
                column_index, row_index, f"{value:+.1f}",
                ha="center", va="center", fontsize=7.2, color=text_color,
            )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02, fraction=0.035)
    colorbar.set_label("Matched residual (pp)")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_100m_term_matched_heatmap")
    write_csv(
        output_dir / "fig_100m_term_matched_heatmap.csv",
        csv_rows,
        ("term", *MATCHED_100M_LABELS, "mean_10M_70M"),
    )


def validate_inputs(behavior: list[dict], dev: list[dict], position: list[dict],
                    contrasts: list[dict]) -> None:
    if len(behavior) != 16470:
        raise RuntimeError(f"unexpected behavior row count: {len(behavior)}")
    if len(dev) != 168:
        raise RuntimeError(f"unexpected dev row count: {len(dev)}")
    if len(position) != 86016:
        raise RuntimeError(f"unexpected position row count: {len(position)}")
    if len(routing_rows(contrasts)) != 13:
        raise RuntimeError("routing contrasts are incomplete")


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[3]
    paper_project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diag-dir", type=Path,
        default=project / "results/paper",
        help="directory containing diag_supp_*.csv inputs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=paper_project / "figures/diagnosis",
        help="destination for individual PNG, SVG, and chart-data CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    behavior = read_csv(args.diag_dir / "diag_supp_behavior_long.csv")
    dev = read_csv(args.diag_dir / "diag_supp_dev_loss.csv")
    position = read_csv(args.diag_dir / "diag_supp_position_nll.csv")
    contrasts = read_csv(args.diag_dir / "diag_supp_masking_contrasts.csv")
    validate_inputs(behavior, dev, position, contrasts)
    configure_style()

    overall_10m = {
        seed: paired_series(behavior, dev, "10m", seed, METRICS["overall_blimp"])
        for seed in SEEDS
    }
    render_schematic(args.output_dir)
    render_trajectory_delta(
        overall_10m, "dev_nll", "Baseline − AttnRes dev NLL (nats)",
        "AttnRes reaches lower dev NLL at equal exposure",
        "19 checkpoints; thin lines are paired seeds; band is the seed range",
        "fig_dev_nll_delta_10m", args.output_dir,
    )
    render_trajectory_delta(
        overall_10m, "blimp", "AttnRes − baseline BLiMP (pp)",
        "Overall BLiMP difference at equal exposure",
        "19 checkpoints; positive values favor AttnRes",
        "fig_dev_blimp_delta_10m", args.output_dir,
    )
    render_matched_blimp(overall_10m, args.output_dir)
    render_position_nll(position, dev, args.output_dir)
    render_routing(
        contrasts, "absolute_ability_pp", r"Unmasked AttnRes ability, $A_U-50$ (pp)",
        "Nonrecent-route dependence is larger on categories with more headroom",
        "n=13 categories; whiskers are five random-mask draws, not training-seed confidence intervals",
        "fig_routing_absolute_ability", args.output_dir,
    )
    render_routing(
        contrasts, "attnres_relative_gain_pp", r"AttnRes − baseline at final checkpoint, $A_U-A_B$ (pp)",
        "Route dependence does not track relative AttnRes gain",
        "n=13 categories; nonrecent cost subtracts count-matched random deletion",
        "fig_routing_relative_gain", args.output_dir,
    )
    render_forest(behavior, dev, args.output_dir)
    render_term_heatmap(behavior, dev, args.output_dir)
    print(f"Rendered figures to {args.output_dir}")


if __name__ == "__main__":
    main()
