from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "outputs" / "icme" / "paper"
FIGURE_ROOT = PAPER_ROOT / "figures"

PARTITIONS = (
    "dirichlet_beta_0.1",
    "dirichlet_beta_0.3",
    "dirichlet_beta_0.5",
    "kmeans",
)
PARTITION_LABELS = ("β=0.1", "β=0.3", "β=0.5", "KMeans")
ESTIMATOR_DATASETS = ("coco", "nus")
ESTIMATOR_LABELS = {"coco": "COCO", "nus": "NUS-WIDE"}
PARTICIPATION_DATASETS = ("voc07", "voc12", "coco", "nus")
DATASET_LABELS = {
    "voc07": "VOC07",
    "voc12": "VOC12",
    "coco": "COCO",
    "nus": "NUS-WIDE",
}
METHOD_STYLES = {
    "Centralized": {
        "label": "Centralized",
        "color": "#4D4D4D",
        "marker": None,
        "linestyle": (0, (1.2, 1.6)),
        "linewidth": 0.95,
        "zorder": 5,
    },
    "Naive-A local-TopK union": {
        "label": "Naive local-union",
        "color": "#D55E00",
        "marker": "o",
        "linestyle": "--",
        "linewidth": 1.35,
        "zorder": 3,
    },
    "FedPASA quantile-aligned": {
        "label": "FedPASA",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 1.55,
        "zorder": 3,
    },
}
PLOT_METHODS = (
    "Naive-A local-TopK union",
    "FedPASA quantile-aligned",
    "Centralized",
)
LEGEND_METHODS = (
    "Centralized",
    "FedPASA quantile-aligned",
    "Naive-A local-TopK union",
)
DATASET_STYLES = {
    "voc07": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "voc12": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "coco": {"color": "#009E73", "marker": "^", "linestyle": "-."},
    "nus": {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
}


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def configure_icme_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.5, alpha=0.72)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def build_figure() -> plt.Figure:
    estimator_rows = load_csv(PAPER_ROOT / "estimator_preserving_summary.csv")
    participation_rows = load_csv(PAPER_ROOT / "client_participation.csv")
    configure_icme_style()

    figure = plt.figure(figsize=(7.16, 2.62))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 2.05), wspace=0.31)
    estimator_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    participation_axis = figure.add_subplot(grid[0, 2])

    method_handles: dict[str, mpl.lines.Line2D] = {}
    for axis, dataset in zip(estimator_axes, ESTIMATOR_DATASETS):
        dataset_values = []
        for method in PLOT_METHODS:
            style = METHOD_STYLES[method]
            lookup = {
                row["partition"]: float(row["mAP_mean"])
                for row in estimator_rows
                if row["dataset"] == dataset and row["method"] == method
            }
            missing = [partition for partition in PARTITIONS if partition not in lookup]
            if missing:
                raise ValueError(f"Missing {dataset}/{method} rows: {missing}")
            values = np.asarray([lookup[partition] for partition in PARTITIONS])
            dataset_values.extend(values.tolist())
            (handle,) = axis.plot(
                np.arange(len(PARTITIONS)),
                values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markerfacecolor="white",
                markeredgewidth=0.95,
                label=style["label"],
                zorder=style["zorder"],
            )
            if dataset == ESTIMATOR_DATASETS[0]:
                method_handles[method] = handle

        padding = max(0.45, 0.10 * (max(dataset_values) - min(dataset_values)))
        axis.set_ylim(min(dataset_values) - padding, max(dataset_values) + padding)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_xticks(np.arange(len(PARTITIONS)), PARTITION_LABELS)
        axis.tick_params(axis="x", pad=2.0)
        axis.set_title(ESTIMATOR_LABELS[dataset], pad=3.0)
        axis.text(
            0.96,
            0.80,
            r"FedPASA $\approx$ Centralized",
            transform=axis.transAxes,
            color="#303030",
            fontsize=6.2,
            ha="right",
            va="center",
        )
        style_axis(axis)

    estimator_axes[0].set_ylabel("mAP (%)")

    participation_handles = []
    for dataset in PARTICIPATION_DATASETS:
        rows = sorted(
            (row for row in participation_rows if row["dataset"] == dataset),
            key=lambda row: float(row["available_clients_percent"]),
        )
        if len(rows) != 4:
            raise ValueError(f"Expected four participation rows for {dataset}, found {len(rows)}")
        style = DATASET_STYLES[dataset]
        container = participation_axis.errorbar(
            [float(row["available_clients_percent"]) for row in rows],
            [float(row["frozen_fusion_mAP_mean"]) for row in rows],
            yerr=[float(row["frozen_fusion_mAP_std"]) for row in rows],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor="white",
            markeredgewidth=0.95,
            capsize=1.8,
            capthick=0.65,
            elinewidth=0.65,
            label=DATASET_LABELS[dataset],
            zorder=3,
        )
        participation_handles.append(container.lines[0])

    participation_axis.set_xlim(20, 104)
    participation_axis.set_ylim(55, 95)
    participation_axis.set_xticks([25, 50, 75, 100])
    participation_axis.set_yticks([60, 70, 80, 90])
    participation_axis.set_xlabel("Available clients (%)")
    participation_axis.set_ylabel("mAP (%)")
    figure.legend(
        handles=participation_handles,
        labels=[DATASET_LABELS[dataset] for dataset in PARTICIPATION_DATASETS],
        loc="upper center",
        bbox_to_anchor=(0.752, 0.910),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=0.55,
    )
    style_axis(participation_axis)

    figure.legend(
        handles=[method_handles[method] for method in LEGEND_METHODS],
        labels=[METHOD_STYLES[method]["label"] for method in LEGEND_METHODS],
        loc="upper center",
        bbox_to_anchor=(0.275, 0.902),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.7,
    )
    figure.text(
        0.275,
        0.978,
        "(a) Pooled-estimator recovery",
        ha="center",
        va="top",
        fontweight="bold",
    )
    figure.text(
        0.752,
        0.978,
        "(b) Client availability",
        ha="center",
        va="top",
        fontweight="bold",
    )
    figure.text(0.275, 0.092, "Partition", ha="center", va="bottom")
    figure.subplots_adjust(left=0.067, right=0.995, bottom=0.235, top=0.76)
    return figure


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_ROOT / "fig3_estimator_participation.pdf"
    figure = build_figure()
    figure.savefig(
        output_path,
        metadata={
            "Title": "FedPASA pooled-estimator recovery and client participation",
            "Subject": "ICME paper Figure 3",
            "Author": "FedPASA",
        },
    )
    plt.close(figure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
