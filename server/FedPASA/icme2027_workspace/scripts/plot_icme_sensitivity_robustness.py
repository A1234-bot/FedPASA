"""Render the ICME-ready sensitivity and robustness figure.

The plot is intentionally generated from the paper JSON artifacts so the figure
is reproducible and does not duplicate experimental values in plotting code.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "outputs" / "icme" / "paper"
FIGURE_ROOT = PAPER_ROOT / "figures"

DATASETS = ("voc07", "voc12", "coco", "nus")
LABELS = {"voc07": "VOC07", "voc12": "VOC12", "coco": "COCO", "nus": "NUS-WIDE"}

# Okabe--Ito inspired colors, paired with independent marker/line encodings so
# the curves remain distinguishable in grayscale and for color-vision deficits.
STYLES = {
    "voc07": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "voc12": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "coco": {"color": "#009E73", "marker": "^", "linestyle": "-."},
    "nus": {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
}


def _load_json(name: str) -> dict:
    path = PAPER_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing input artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_icme_style() -> None:
    """Use compact, print-safe typography for an IEEE two-column paper."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 9.1,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure() -> plt.Figure:
    temperature = _load_json("temperature_sensitivity.json")
    participation = _load_json("client_participation.json")
    verification = _load_json("verification_report.json")

    _configure_icme_style()
    # 7.16 in fits the usable width of the standard two-column IEEE template.
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.60))
    ax_temp, ax_client = axes

    handles = []
    temperature_ranges = []
    for dataset in DATASETS:
        rows = temperature[dataset]["rows"]
        style = STYLES[dataset]
        reference_mAP = next(row["mAP"] for row in rows if row["predeclared_reference_multiplier"])
        delta_mAP = np.asarray([row["mAP"] - reference_mAP for row in rows], dtype=float)
        temperature_ranges.append(float(np.ptp(delta_mAP)))
        (line,) = ax_temp.plot(
            [row["temperature_multiplier"] for row in rows],
            delta_mAP,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor="white",
            markeredgewidth=1.05,
            label=LABELS[dataset],
            zorder=3,
        )
        handles.append(line)

    ax_temp.axhline(
        0.0,
        color="#969696",
        linewidth=0.85,
        linestyle=(0, (3, 2)),
        alpha=0.95,
        zorder=4,
    )
    ax_temp.set_xscale("log", base=2)
    ax_temp.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax_temp.set_xticklabels([r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$", r"$2^{1}$", r"$2^{2}$"])
    ax_temp.set_xlabel(r"Temperature multiplier $\tau/\tau_0$")
    ax_temp.set_ylabel(r"$\Delta$mAP (%)")
    ax_temp.set_ylim(-0.05, 0.05)
    ax_temp.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
    ax_temp.set_yticklabels([r"$-0.05$", r"$-0.025$", "$0$", r"$+0.025$", r"$+0.05$"])
    ax_temp.set_title("(a) Temperature Sensitivity", loc="left", pad=4.0)
    ax_temp.text(
        0.98,
        0.93,
        rf"Max variation: ${max(temperature_ranges):.2f}$ mAP",
        transform=ax_temp.transAxes,
        ha="right",
        va="top",
        color="#3F3F3F",
        fontsize=8.0,
    )
    _style_axis(ax_temp)

    full_to_75_drops = []
    for dataset in DATASETS:
        summary_by_percent = {
            float(row["available_clients_percent"]): row for row in participation[dataset]["summary"]
        }
        style = STYLES[dataset]
        x = np.asarray([25.0, 50.0, 75.0, 100.0])
        y = np.asarray(
            [
                summary_by_percent[25.0]["frozen_fusion_mAP_mean"],
                summary_by_percent[50.0]["frozen_fusion_mAP_mean"],
                summary_by_percent[75.0]["frozen_fusion_mAP_mean"],
                verification["datasets"][dataset]["cv_mAP"],
            ],
            dtype=float,
        )
        yerr = np.asarray(
            [
                summary_by_percent[25.0]["frozen_fusion_mAP_std"],
                summary_by_percent[50.0]["frozen_fusion_mAP_std"],
                summary_by_percent[75.0]["frozen_fusion_mAP_std"],
                0.0,
            ],
            dtype=float,
        )
        full_to_75_drops.append(float(y[-1] - y[2]))
        ax_client.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor="white",
            markeredgewidth=1.05,
            capsize=2.0,
            capthick=0.75,
            elinewidth=0.75,
            zorder=3,
        )

    ax_client.set_xlabel("Client participation (%)")
    ax_client.set_ylabel("mAP (%)")
    ax_client.set_xticks([25, 50, 75, 100])
    ax_client.set_xlim(20, 104)
    ax_client.set_ylim(55.0, 95.0)
    ax_client.set_yticks([60, 70, 80, 90])
    ax_client.set_title("(b) Client Availability Robustness", loc="left", pad=4.0)
    ax_client.axvspan(20.0, 35.0, color="#D9D9D9", alpha=0.38, linewidth=0.0, zorder=0)
    ax_client.text(
        0.98,
        0.21,
        rf"Max loss @75%: ${max(full_to_75_drops):.2f}$ mAP",
        transform=ax_client.transAxes,
        ha="right",
        va="bottom",
        color="#3F3F3F",
        fontsize=8.0,
    )
    ax_client.text(
        31.5,
        83.0,
        "Low-participation\nregime",
        ha="center",
        va="center",
        color="#555555",
        fontsize=7.6,
        linespacing=0.90,
    )
    _style_axis(ax_client)

    figure.legend(
        handles=handles,
        labels=[LABELS[dataset] for dataset in DATASETS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.5,
    )
    figure.subplots_adjust(left=0.105, right=0.995, bottom=0.205, top=0.80, wspace=0.30)
    return figure


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    png_path = FIGURE_ROOT / "sensitivity_robustness.png"
    pdf_path = FIGURE_ROOT / "sensitivity_robustness.pdf"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path)
    plt.close(figure)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
