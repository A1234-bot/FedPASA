"""Rebuild the paper figure from saved AP values, without rerunning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs" / "icme" / "paper"


def _plot_complementarity(report: dict, figure_root: Path, *, single_column: bool) -> None:
    figure_root = figure_root or PAPER / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    datasets = [name for name in ("coco", "nus") if name in report["datasets"]]
    if not datasets:
        datasets = list(report["datasets"])[-2:]
    if not datasets:
        raise ValueError("No classwise AP data available")

    # Fail explicitly if no Times-compatible font exists; never silently use sans.
    for family in ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("Install a Times-compatible font before exporting the figure")

    font_size = 9 if single_column else 10
    title_size = 10
    inset_size = 10
    style = {
        "font.family": family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": title_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "savefig.facecolor": "white",
        "savefig.bbox": None,
    }
    with plt.rc_context(style):
        # Fixed physical size: no tight bounding-box crop that changes print scale.
        width_mm, height_mm = (86, 108) if single_column else (178, 96)
        figure, axes = plt.subplots(
            len(datasets), 1, figsize=(width_mm / 25.4, height_mm / 25.4),
            sharex=True, squeeze=False,
        )
        figure.subplots_adjust(
            left=0.155 if single_column else 0.09, right=0.98,
            bottom=0.115, top=0.975, hspace=0.18,
        )
        largest_index = max(
            row["class_id"] for dataset in datasets
            for row in report["datasets"][dataset]["per_class"]
        )
        names = {"coco": "COCO", "nus": "NUS-WIDE"}
        for panel, (axis, dataset) in enumerate(zip(axes[:, 0], datasets)):
            rows = report["datasets"][dataset]["per_class"]
            indices = np.asarray([row["class_id"] for row in rows])
            # Source AP is already in percent; subtraction yields percentage points.
            differences = np.asarray([row["full_minus_ridge_AP"] for row in rows])
            negative = differences < 0
            axis.bar(
                indices, differences, width=0.82,
                color=np.where(negative, "#C44E52", "#2F6B9A"),
                edgecolor=np.where(negative, "#63272B", "#2F6B9A"),
                linewidth=np.where(negative, 0.4, 0.0), zorder=3,
            )
            low, high = min(0.0, differences.min()), max(0.0, differences.max())
            span = max(high - low, 1.0)
            axis.set_ylim(np.floor(low - 0.08 * span), high + 0.18 * span)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 4, 5, 10]))
            # Draw only nonzero grid lines; the zero baseline has its own weight.
            for tick in axis.get_yticks():
                if axis.get_ylim()[0] <= tick <= axis.get_ylim()[1] and not np.isclose(tick, 0):
                    axis.axhline(tick, color="#EEEEEE", linewidth=0.4, zorder=0)
            axis.axhline(0, color="#333333", linewidth=1.0, zorder=4)
            axis.set_ylabel("ΔAP (pp)", labelpad=5)
            axis.set_xlim(-1, largest_index + 1)
            axis.text(
                0.014, 0.945, f"({chr(97 + panel)}) {names.get(dataset, dataset.upper())}",
                transform=axis.transAxes, ha="left", va="top",
                fontsize=title_size, fontweight="bold",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", labelbottom=panel == len(datasets) - 1)
            if dataset == "coco":
                # Occupy the empty region above ordinary bars, leaving the full
                # 12 pp peak unobscured in the main axes. Main bars are unchanged.
                bounds = [0.20, 0.43, 0.58, 0.40] if single_column else [0.20, 0.43, 0.60, 0.43]
                # Shift the complete inset right by 3 physical points in either layout.
                bounds[0] += 3.0 / (figure.get_figwidth() * 72 * axis.get_position().width)
                zoom = axis.inset_axes(bounds, zorder=5)
                zoom.set_facecolor("white")
                zoom.bar(
                    indices, differences, width=0.82,
                    color=np.where(negative, "#C44E52", "#2F6B9A"),
                    edgecolor=np.where(negative, "#63272B", "#2F6B9A"),
                    linewidth=np.where(negative, 0.4, 0.0), zorder=3,
                )
                zoom.set_xlim(axis.get_xlim())
                zoom.set_ylim(-0.5, 2.5)
                zoom.set_xticks([0, 40, 80])
                zoom.set_yticks([-0.5, 1.0, 2.5])
                zoom.tick_params(labelsize=inset_size, length=2, width=0.5, pad=1.5)
                zoom.set_title("Zoomed view", fontsize=inset_size, pad=6.5)
                zoom.axhline(0, color="#333333", linewidth=1.0, zorder=4)
                zoom.axhline(1.0, color="#EEEEEE", linewidth=0.4, zorder=0)
                for spine in zoom.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_color("#999999")
                # Explicitly identify bars exceeding the inset range; the main
                # panel always displays their true heights without truncation.
                overflow = differences > 2.5
                zoom.scatter(
                    indices[overflow], np.full(overflow.sum(), 2.5),
                    marker="^", s=12 if single_column else 16,
                    color="#2F6B9A", edgecolors="white", linewidths=0.35,
                    clip_on=False, zorder=6,
                )
        axes[-1, 0].set_xticks(np.arange(0, largest_index + 1, 10))
        axes[-1, 0].set_xlabel("Class Index", labelpad=4)
        stem = "classwise_complementarity" + ("_single_column" if single_column else "")
        figure.savefig(figure_root / f"{stem}.pdf")
        figure.savefig(figure_root / f"{stem}.png", dpi=600)
        plt.close(figure)


def plot_complementarity(report: dict, figure_root: Path | None = None) -> None:
    """Export separately typeset double- and single-column figures."""
    for single_column in (False, True):
        _plot_complementarity(report, figure_root or PAPER / "figures", single_column=single_column)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PAPER / "classwise_complementarity.json")
    parser.add_argument("--output-dir", type=Path, default=PAPER / "figures")
    args = parser.parse_args()
    plot_complementarity(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()
