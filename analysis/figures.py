#!/usr/bin/env python3
"""Render the paper diagnostic figures from archived numerical evidence.

The numerical transformations are imported from the paper's original plotting
script so that the samples differ only in presentation, not in analysis.
PDF is the publication output.  A PNG contact sheet is generated solely for
quick visual review.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
SOURCE_SCRIPT = ROOT / "analysis/make_diagnostic_figures.py"
DEFAULT_DIAG = ROOT / "results/paper"
DEFAULT_OUTPUT = ROOT / "outputs/figures"

spec = importlib.util.spec_from_file_location("paper_figures", SOURCE_SCRIPT)
paper_figures = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(paper_figures)

SEEDS = paper_figures.SEEDS
POSITION_BINS = paper_figures.POSITION_BINS
MATCHED_100M_LABELS = paper_figures.MATCHED_100M_LABELS

COLORS = {
    "blue": "#2166AC",
    "blue_light": "#A9C7E2",
    "red": "#B2182B",
    "ink": "#222222",
    "gray": "#6B6B6B",
    "light_gray": "#C8C8C8",
    "grid": "#E1E1E1",
    "paper": "#FFFFFF",
}


def configure_style() -> None:
    """Match the paper's Times text and keep vector text in PDF."""
    plt.rcParams.update({
        "text.usetex": False,
        "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": COLORS["ink"],
        "axes.linewidth": 0.65,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": COLORS["paper"],
    })


def style_axis(axis, *, xlog: bool = False, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid_axis, color=COLORS["grid"], lw=0.55, zorder=0)
    axis.tick_params(width=0.65, length=3)
    if xlog:
        axis.set_xscale("log")
        ticks = np.asarray([1, 2, 5, 10, 20, 30, 50, 90]) * 1e6
        axis.set_xticks(ticks, ["1", "2", "5", "10", "20", "30", "50", "90"])
        axis.minorticks_off()
        axis.set_xlabel("Words seen (M)")


def save_pdf(fig, output_dir: Path, stem: str, *, crop: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02} if crop else {}
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    return path


def save_development_pdf(fig, output_dir: Path, stem: str) -> Path:
    """Use one fixed canvas and axes rectangle for all three dev panels."""
    fig.subplots_adjust(left=0.255, right=0.985, bottom=0.235, top=0.98)
    return save_pdf(fig, output_dir, stem, crop=False)


def load_inputs(diag_dir: Path):
    behavior = paper_figures.read_csv(diag_dir / "diag_supp_behavior_long.csv")
    dev = paper_figures.read_csv(diag_dir / "diag_supp_dev_loss.csv")
    position = paper_figures.read_csv(diag_dir / "diag_supp_position_nll.csv")
    contrasts = paper_figures.read_csv(diag_dir / "diag_supp_masking_contrasts.csv")
    paper_figures.validate_inputs(behavior, dev, position, contrasts)
    overall_10m = {
        seed: paper_figures.paired_series(
            behavior, dev, "10m", seed, paper_figures.METRICS["overall_blimp"]
        )
        for seed in SEEDS
    }
    return behavior, dev, position, contrasts, overall_10m


def render_trajectory(series_by_seed: dict, field: str, ylabel: str,
                      stem: str, output_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(3.15, 2.25))
    values, words = [], []
    for seed in SEEDS:
        series = series_by_seed[seed]
        delta = (series["baseline_nll"] - series["attnres_nll"]
                 if field == "dev_nll"
                 else series["attnres_accuracy"] - series["baseline_accuracy"])
        values.append(delta)
        words.append(series["words"])
        # Every checkpoint uses the same circular marker and size; the dashed
        # vertical reference alone carries the warmup boundary.
        axis.plot(series["words"], delta, color=COLORS["blue"], alpha=0.28,
                  lw=0.75, marker="o", ms=2.4, zorder=2)
    values = np.asarray(values)
    mean_words = np.mean(words, axis=0)
    axis.fill_between(mean_words, values.min(axis=0), values.max(axis=0),
                      color=COLORS["blue"], alpha=0.10, lw=0, zorder=1)
    axis.plot(mean_words, values.mean(axis=0), color=COLORS["blue"], lw=1.8,
              label="Mean of 3 paired seeds", zorder=4)
    axis.axhline(0, color=COLORS["gray"], lw=0.65, ls=(0, (3, 2)), zorder=1)
    axis.axvline(8e6, color=COLORS["gray"], lw=0.65, ls=(0, (3, 2)), zorder=1)
    style_axis(axis, xlog=True)
    axis.set_ylabel(ylabel)
    axis.legend(loc="upper left")
    return save_development_pdf(fig, output_dir, stem)


def render_matched(series_by_seed: dict, output_dir: Path) -> Path:
    matched = {seed: paper_figures.matched_delta(series_by_seed[seed]) for seed in SEEDS}
    common = [label for label in series_by_seed[SEEDS[0]]["labels"]
              if all(label in matched[seed] for seed in SEEDS)]
    fig, axis = plt.subplots(figsize=(3.15, 2.25))
    for seed in SEEDS:
        labels = [label for label in series_by_seed[seed]["labels"] if label in matched[seed]]
        x = [matched[seed][label]["words"] for label in labels]
        y = [matched[seed][label]["delta"] for label in labels]
        axis.plot(x, y, color=COLORS["blue"], alpha=0.28, lw=0.75,
                  marker="o", ms=2.4, zorder=2)
    common_x = np.asarray([
        np.mean([matched[seed][label]["words"] for seed in SEEDS]) for label in common
    ])
    common_y = np.asarray([
        [matched[seed][label]["delta"] for label in common] for seed in SEEDS
    ]).T
    axis.fill_between(common_x, common_y.min(axis=1), common_y.max(axis=1),
                      color=COLORS["blue"], alpha=0.10, lw=0)
    axis.plot(common_x, common_y.mean(axis=1), color=COLORS["blue"], lw=1.8,
              label="Mean where all seeds overlap", zorder=4)
    axis.axhline(0, color=COLORS["gray"], lw=0.65, ls=(0, (3, 2)))
    axis.axvline(8e6, color=COLORS["gray"], lw=0.65, ls=(0, (3, 2)))
    style_axis(axis, xlog=True)
    axis.set_ylabel("Matched BLiMP residual (pp)")
    axis.legend(loc="upper left")
    return save_development_pdf(fig, output_dir, "fig_dev_blimp_matched_10m")


def render_position(position: list[dict], dev: list[dict], output_dir: Path) -> Path:
    pidx = paper_figures.unique_index(
        position,
        ("corpus", "seed", "architecture", "checkpoint_label", "loss_index"),
        "position",
    )
    didx = paper_figures.dev_index(dev)
    curves = {}
    for seed in SEEDS:
        contexts, deltas, counts = [], [], []
        for loss_index in range(512):
            baseline = pidx[("10m", str(seed), "baseline", "final", str(loss_index))]
            attnres = pidx[("10m", str(seed), "attnres", "final", str(loss_index))]
            contexts.append(int(baseline["context_length"]))
            counts.append(int(baseline["token_count"]))
            deltas.append(float(baseline["mean_nll"]) - float(attnres["mean_nll"]))
        overall = (float(didx[("10m", str(seed), "baseline", "final")]["mean_nll"])
                   - float(didx[("10m", str(seed), "attnres", "final")]["mean_nll"]))
        curves[seed] = {
            "bins": paper_figures.weighted_position_bins(contexts, deltas, counts),
            "overall": overall,
        }
    fig, axis = plt.subplots(figsize=(3.35, 2.35))
    seed_values = []
    for seed in SEEDS:
        x = [row["x"] for row in curves[seed]["bins"]]
        y = [row["delta"] for row in curves[seed]["bins"]]
        seed_values.append(y)
        axis.plot(x, y, color=COLORS["blue"], alpha=0.28, lw=0.75,
                  marker="o", ms=2.6)
    axis.plot(x, np.mean(seed_values, axis=0), color=COLORS["blue"], lw=1.8,
              marker="o", ms=3.4, label="Mean of 3 paired seeds")
    overall = float(np.mean([curves[seed]["overall"] for seed in SEEDS]))
    axis.axhline(overall, color=COLORS["gray"], lw=0.75, ls=(0, (4, 2)))
    axis.annotate(
        rf"Token-weighted overall: {overall:.3f}",
        xy=(0.99, overall), xycoords=("axes fraction", "data"),
        xytext=(0, -3), textcoords="offset points", ha="right", va="top",
        fontsize=7.2, color=COLORS["gray"],
    )
    axis.axhline(0, color=COLORS["gray"], lw=0.65, ls=(0, (3, 2)))
    axis.set_ylim(-0.004, 0.038)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [f"{a}--{b}" for a, b in POSITION_BINS])
    axis.minorticks_off()
    axis.set_xlabel("Context-position bin")
    axis.set_ylabel(r"Baseline $-$ AttnRes dev NLL (nats)")
    style_axis(axis)
    # Center the legend at 0.0025 nats: exactly midway between the dashed
    # zero reference and the 0.005 major grid line, without touching either.
    legend_y = (0.0025 - axis.get_ylim()[0]) / np.diff(axis.get_ylim())[0]
    axis.legend(loc="center left", bbox_to_anchor=(0.0, legend_y))
    fig.tight_layout(pad=0.45)
    return save_pdf(fig, output_dir, "fig_position_nll_final_10m")


def heatmap_matrix(behavior: list[dict], dev: list[dict]):
    terms = list(paper_figures.term_specs(behavior))
    matrix = np.empty((len(terms), len(MATCHED_100M_LABELS)), dtype=np.float64)
    for term_index, (term, metric_spec) in enumerate(paper_figures.term_specs(behavior).items()):
        seed_matches = {
            seed: paper_figures.matched_delta(
                paper_figures.paired_series(behavior, dev, "100m", seed, metric_spec)
            )
            for seed in SEEDS
        }
        for label_index, label in enumerate(MATCHED_100M_LABELS):
            matrix[term_index, label_index] = np.mean(
                [seed_matches[seed][label]["delta"] for seed in SEEDS]
            )
    return terms, np.column_stack([matrix, matrix.mean(axis=1)])


def render_heatmap(behavior: list[dict], dev: list[dict], output_dir: Path) -> Path:
    terms, display = heatmap_matrix(behavior, dev)
    extent = float(np.max(np.abs(display)))
    # Signed semantic mapping: negative red, zero near-white, positive blue.
    cmap = LinearSegmentedColormap.from_list(
        "signed_red_blue", (COLORS["red"], "#FAFAFA", COLORS["blue"])
    )
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    fig, axis = plt.subplots(figsize=(6.65, 4.25))
    image = axis.imshow(display, cmap=cmap, norm=norm, aspect="auto")
    xlabels = [*MATCHED_100M_LABELS, "Mean"]
    axis.set_xticks(np.arange(len(xlabels)), xlabels)
    axis.set_yticks(np.arange(len(terms)), [term.replace("_", " ") for term in terms])
    axis.set_xlabel("AttnRes checkpoint (words seen)")
    axis.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(terms), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.85)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.axvline(len(MATCHED_100M_LABELS) - 0.5, color=COLORS["ink"], lw=1.0)
    for row_index in range(display.shape[0]):
        for column_index in range(display.shape[1]):
            value = display[row_index, column_index]
            color = "white" if abs(value) > 0.52 * extent else COLORS["ink"]
            axis.text(column_index, row_index, f"{value:+.1f}", ha="center", va="center",
                      fontsize=6.9, color=color)
    colorbar = fig.colorbar(image, ax=axis, pad=0.015, fraction=0.032)
    colorbar.set_label("Matched residual (pp)")
    fig.tight_layout(pad=0.4)
    return save_pdf(fig, output_dir, "fig_100m_term_matched_heatmap")


def routing_arrays(contrasts: list[dict], x_field: str):
    rows = paper_figures.routing_rows(contrasts)
    x = np.asarray([float(row[x_field]) for row in rows])
    y = np.asarray([float(row["nonrecent_excess_cost_pp"]) for row in rows])
    low = np.asarray([float(row["draw_aggregate_min_pp"]) for row in rows])
    high = np.asarray([float(row["draw_aggregate_max_pp"]) for row in rows])
    return x, y, low, high


def draw_routing(axis, contrasts: list[dict], x_field: str, xlabel: str,
                 *, show_ylabel: bool) -> None:
    x, y, low, high = routing_arrays(contrasts, x_field)
    axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="o",
                  color=COLORS["blue"], ecolor=COLORS["light_gray"],
                  elinewidth=0.8, capsize=2.0, ms=3.5, zorder=3)
    axis.axhline(0, color=COLORS["gray"], lw=0.65, ls=(0, (3, 2)))
    axis.axvline(0, color=COLORS["gray"], lw=0.65, ls=(0, (3, 2)))
    axis.set_xlabel(xlabel)
    if show_ylabel:
        axis.set_ylabel(r"Nonrecent masking cost $\bar{A}_R-A_N$ (pp)")
    axis.set_ylim(-2.1, 11.4)
    style_axis(axis)


def render_routing(contrasts: list[dict], output_dir: Path) -> tuple[Path, Path]:
    # Keep the panels as separate, identically sized PDFs. Fixed margins and
    # uncropped export guarantee the same physical page and axes proportions.
    panel_specs = (
        ("absolute_ability_pp", r"Unmasked AttnRes ability, $A_U-50$ (pp)",
         "fig_routing_absolute_ability"),
        ("attnres_relative_gain_pp",
         r"AttnRes $-$ baseline at final checkpoint, $A_U-A_B$ (pp)",
         "fig_routing_relative_gain"),
    )
    paths = []
    for x_field, xlabel, stem in panel_specs:
        fig, axis = plt.subplots(figsize=(3.35, 2.35))
        draw_routing(axis, contrasts, x_field, xlabel, show_ylabel=True)
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.98)
        paths.append(save_pdf(fig, output_dir, stem, crop=False))
    return paths[0], paths[1]


def rasterize_for_preview(pdf_path: Path, png_path: Path, dpi: int = 150) -> None:
    from subprocess import run
    stem = png_path.with_suffix("")
    run(["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", str(dpi),
         str(pdf_path), str(stem)], check=True)


def make_contact_sheet(pdf_paths: list[Path], output_dir: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    selected = [
        next(path for path in pdf_paths if path.stem == "fig_position_nll_final_10m"),
        next(path for path in pdf_paths if path.stem == "fig_100m_term_matched_heatmap"),
    ]
    rendered = []
    for pdf in selected:
        png = output_dir / f"preview_{pdf.stem}.png"
        rasterize_for_preview(pdf, png)
        rendered.append((pdf.stem, Image.open(png).convert("RGB")))
    width = 1700
    margin, gap, label_height = 42, 35, 38
    scaled = []
    for name, image in rendered:
        target_width = width - 2 * margin
        height = round(image.height * target_width / image.width)
        scaled.append((name, image.resize((target_width, height), Image.Resampling.LANCZOS)))
    total_height = 2 * margin + sum(label_height + image.height for _, image in scaled) + gap * (len(scaled) - 1)
    sheet = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=25)
    y = margin
    for index, (name, image) in enumerate(scaled):
        draw.text((margin, y), f"Sample {index + 1}: {name}", fill=COLORS["ink"], font=font)
        y += label_height
        sheet.paste(image, (margin, y))
        y += image.height + gap
    path = output_dir / "figure_revision_contact_sheet.png"
    sheet.save(path, dpi=(150, 150))
    return path


def make_row_preview(pdf_paths: list[Path], output_dir: Path, *,
                     stems: tuple[str, ...], labels: tuple[str, ...],
                     filename: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    rendered = []
    for stem in stems:
        pdf = next(path for path in pdf_paths if path.stem == stem)
        png = output_dir / f"preview_{stem}.png"
        rasterize_for_preview(pdf, png, dpi=180)
        rendered.append(Image.open(png).convert("RGB"))
    margin, gap, label_height = 24, 24, 36
    cell_width = 650
    scaled = []
    for source in rendered:
        height = round(source.height * cell_width / source.width)
        scaled.append(source.resize((cell_width, height), Image.Resampling.LANCZOS))
    canvas_width = 2 * margin + len(scaled) * cell_width + (len(scaled) - 1) * gap
    canvas_height = 2 * margin + label_height + max(image.height for image in scaled)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=25)
    for index, (label, image) in enumerate(zip(labels, scaled)):
        x = margin + index * (cell_width + gap)
        draw.text((x, margin), label, fill=COLORS["ink"], font=font)
        canvas.paste(image, (x, margin + label_height))
    path = output_dir / filename
    canvas.save(path, dpi=(150, 150))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-previews", action="store_true", help="Write the seven PDFs without requiring Poppler")
    args = parser.parse_args()
    configure_style()
    behavior, dev, position, contrasts, overall_10m = load_inputs(args.diag_dir)
    paths = [
        render_position(position, dev, args.output_dir),
        render_trajectory(overall_10m, "dev_nll",
                          r"Baseline $-$ AttnRes dev NLL (nats)",
                          "fig_dev_nll_delta_10m", args.output_dir),
        render_trajectory(overall_10m, "blimp",
                          r"AttnRes $-$ baseline BLiMP (pp)",
                          "fig_dev_blimp_delta_10m", args.output_dir),
        render_matched(overall_10m, args.output_dir),
        render_heatmap(behavior, dev, args.output_dir),
    ]
    paths.extend(render_routing(contrasts, args.output_dir))
    if args.skip_previews:
        print("\n".join(map(str, paths)))
        return
    contact = make_contact_sheet(paths, args.output_dir)
    developmental = make_row_preview(
        paths, args.output_dir,
        stems=("fig_dev_nll_delta_10m", "fig_dev_blimp_delta_10m",
               "fig_dev_blimp_matched_10m"),
        labels=("(a) Equal-exposure dev NLL", "(b) Equal-exposure BLiMP",
                "(c) Matched-NLL BLiMP"),
        filename="developmental_10m_preview.png",
    )
    routing = make_row_preview(
        paths, args.output_dir,
        stems=("fig_routing_absolute_ability", "fig_routing_relative_gain"),
        labels=("(a) Absolute ability", "(b) Relative gain over baseline"),
        filename="routing_relationships_preview.png",
    )
    print("\n".join(str(path) for path in [*paths, contact, developmental, routing]))


if __name__ == "__main__":
    main()
