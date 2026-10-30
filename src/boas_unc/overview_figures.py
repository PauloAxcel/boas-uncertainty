"""Overview figures for the Phase 0 results (human vs the two AI label sources).

Numbers come from results/agreement.json and results/hotspots.json (run boas_unc.agreement and
boas_unc.hotspots first); the example
hypnograms also read the label files. Writes figures/overview/*.png.

Run from the repo root:  uv run python -m boas_unc.overview_figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from boas_unc.inventory_figures import (COLORS, GRID, HYP_LABELS, HYP_Y, INK, INK2, MUTED, SEQ_BLUE,
                                        SURFACE, style)
from boas_unc.load import STAGE_LABELS, load_all, runs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
AGREEMENT = ROOT / "results" / "agreement.json"
HOTSPOTS = ROOT / "results" / "hotspots.json"
FIG_DIR = ROOT / "figures" / "overview"

STAGES = list(STAGE_LABELS.values())


def fig_confusion(agreement: dict, out: Path):
    """Row-normalised: when the human says X, what does the headband say? (end-to-end version)"""
    pooled = agreement["pairs"]["human_vs_hb_ai"]["b_end_to_end"]["pooled"]
    cols = STAGES + ["-2"]
    norm = pd.DataFrame(pooled["confusion_row_normalised"]).T.reindex(index=STAGES, columns=cols)
    counts = pd.DataFrame(pooled["confusion_counts"]).T.reindex(index=STAGES, columns=cols)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.imshow(norm.to_numpy() * 100, cmap=cmap, vmin=0, vmax=100)
    for (r, c), v in np.ndenumerate(norm.to_numpy() * 100):
        ax.text(c, r, f"{v:.0f}%" if v >= 0.5 else "<1%" if v > 0 else "0", ha="center", va="center",
                fontsize=10, fontweight="bold" if r == c else "normal", color="#ffffff" if v > 45 else INK)
        ax.text(c, r + 0.3, f"{int(counts.iat[r, c]):,}".replace(",", " "), ha="center", va="center",
                fontsize=6.5, color="#ffffff" if v > 45 else INK2)
    ax.set_xticks(range(len(cols)), cols[:-1] + ["no stage\n(-2)"])
    ax.set_yticks(range(len(STAGES)), [f"{s}\n(n={int(counts.loc[s].sum()):,})".replace(",", " ") for s in STAGES])
    ax.set_xlabel("Headband AI says…")
    ax.set_ylabel("Human says…")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("When the human scores a stage, what does the headband say?")
    fig.text(0.01, 0.005, "Each row sums to 100 %. Diagonal = agreement. Small numbers = epochs.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_stage_f1(agreement: dict, out: Path):
    """Per-stage F1, pooled over all epochs where both labels are 0-4, human as the reference."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.34
    for i, (pair, col, name) in enumerate([("human_vs_psg_ai", "psg_ai", "AI on PSG"),
                                           ("human_vs_hb_ai", "hb_ai", "AI on headband")]):
        per_stage = agreement["pairs"][pair]["a_both_stage"]["pooled"]["per_stage"]
        f1 = [100 * per_stage[s]["f1"] for s in STAGES]
        x = np.arange(len(STAGES)) + (i - 0.5) * (width + 0.04)
        ax.bar(x, f1, width, color=COLORS[col], label=name)
        for xi, v in zip(x, f1):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(STAGES)), STAGES)
    ax.set_ylim(0, 105)
    ax.set_ylabel("F1 score (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2, ncol=2)
    ax.set_title("How well each AI finds each stage (human = reference)")
    fig.text(0.01, 0.005, "F1 balances precision and recall: 100 = perfect, 0 = never right. "
             "Epochs where both labels are 0-4.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_per_night_accuracy(agreement: dict, out: Path):
    """Accuracy of each of the 128 nights: one dot per night, so the spread is visible."""
    pairs = agreement["pairs"]
    panels = [("Human vs AI on PSG", "psg_ai", pairs["human_vs_psg_ai"]["a_both_stage"]),
              ("Human vs headband\n(staging only)", "hb_ai", pairs["human_vs_hb_ai"]["a_both_stage"]),
              ("Human vs headband\n(dropout counted\nas an error)", "hb_ai", pairs["human_vs_hb_ai"]["b_end_to_end"])]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    gap = 1.7  # horizontal distance between columns, leaves room for the median label
    for k, (name, col, block) in enumerate(panels):
        i = k * gap
        acc = pd.Series(block["per_night_accuracy_by_night"]).astype(float) * 100
        q25, med, q75 = np.percentile(acc, [25, 50, 75])
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(acc)), acc, s=14, color=COLORS[col], alpha=0.6,
                   edgecolors="none")
        ax.hlines(med, i - 0.3, i + 0.3, color=INK, lw=2)
        ax.vlines(i + 0.33, q25, q75, color=INK2, lw=2)
        ax.text(i + 0.38, med, f"median {med:.1f}%\nIQR {q25:.1f}–{q75:.1f}", va="center", fontsize=8, color=INK2)
        worst = acc.idxmin()
        ax.annotate(f"{worst}: {acc.min():.0f}%", (i, acc.min()), xytext=(i - 0.05, acc.min() - 3.5),
                    fontsize=7.5, color=MUTED, ha="center")
    ax.set_xticks([k * gap for k in range(len(panels))], [p[0] for p in panels])
    ax.set_xlim(-0.5, (len(panels) - 1) * gap + 1.2)
    ax.set_ylabel("Accuracy of one night (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Nights differ a lot: one dot = one night (128 per column)")
    fig.text(0.01, 0.005, "Bar = median, vertical line = middle 50 % of nights (IQR). "
             "Lowest night is labelled.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pick_nights(agreement: dict) -> dict:
    """A typical night (accuracy closest to the median) and the lowest-accuracy night, headband end-to-end."""
    acc = pd.Series(agreement["pairs"]["human_vs_hb_ai"]["b_end_to_end"]["per_night_accuracy_by_night"]).astype(float)
    return {"typical": (acc - acc.median()).abs().idxmin(), "lowest": acc.idxmin()}


def fig_example_nights(table: pd.DataFrame, agreement: dict, out: Path):
    """Human and headband hypnograms side by side for a typical and for the worst night."""
    picked = pick_nights(agreement)
    acc = pd.Series(agreement["pairs"]["human_vs_hb_ai"]["b_end_to_end"]["per_night_accuracy_by_night"]).astype(float)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.2))
    step = 30 / 3600
    for block, (kind, sub) in enumerate(picked.items()):
        night = table[table["sub"] == sub].sort_values("onset")
        hours = night.onset.to_numpy() / 3600
        for j, (col, name) in enumerate([("psg_hum", "Human"), ("hb_ai", "Headband AI")]):
            ax = axes[2 * block + j]
            codes = night[col].to_numpy()
            y = np.array([HYP_Y.get(c, np.nan) for c in codes], dtype=float)
            ax.step(np.append(hours, hours[-1] + step), np.append(y, y[-1]), where="post",
                    color=COLORS[col], lw=1.4)
            for code in (8, -2):
                for start, length in runs(codes, code):
                    ax.axvspan(hours[start], hours[start] + length * step, fc=MUTED, ec=MUTED, alpha=0.35, lw=0.8)
            ax.set_yticks(range(5), HYP_LABELS)
            ax.set_ylim(-0.5, 4.5)
            ax.set_xlim(0, hours[-1] + step)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_ylabel(name, fontsize=9)
            if j == 0:
                ax.set_title(f"{'Typical' if kind == 'typical' else 'Lowest-accuracy'} night: {sub}, "
                             f"headband accuracy {100 * acc[sub]:.0f} %", fontsize=10)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Hours since recording start")
    fig.text(0.01, 0.005, "Grey bands = no stage (human 8: PSG disconnected; headband -2: artefact / missing).",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_dropout_by_night(hotspots: dict, out: Path):
    """Share of human-scored epochs where the headband gives no stage (-2), one bar per night."""
    rates = (pd.Series(hotspots["per_night_headband_missing_rate"]).astype(float) * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(range(len(rates)), rates.to_numpy(), width=0.8, color=COLORS["hb_ai"])
    ax.axhline(20, color=INK2, lw=0.8)
    ax.text(len(rates) - 1, 21, "20 %: nights above this are left out of the matched-epoch analysis (D3)",
            ha="right", fontsize=8, color=INK2)
    for i, (sub, v) in enumerate(rates[rates > 20].items()):
        ax.text(i + 0.4, v + 1, sub, fontsize=8, color=INK2, ha="left")
    n_any = int((rates > 0).sum())
    ax.set_xlim(-1, len(rates))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(rates)} nights, sorted from most to least dropout ({n_any} have any; "
                  f"median {rates.median():.2f} %)")
    ax.set_ylabel("Headband gives no stage (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Headband dropout is concentrated in a few nights")
    fig.text(0.01, 0.005, "Share of the epochs the human scored 0-4 where the headband AI says -2 (artefact / missing).",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_groups(hotspots: dict, out: Path):
    """Headband disagreement by age band and by sex: per-person median and middle 50 %."""
    overall = 100 * hotspots["per_participant"]["rate_summary"]["median"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True, gridspec_kw={"width_ratios": [4, 2.2]})
    for ax, (title, key) in zip(axes, [("Age band (years)", "by_age_band"), ("Sex", "by_sex")]):
        groups = hotspots[key]
        for i, (name, g) in enumerate(groups.items()):
            r = g["per_person_rate"]
            ax.vlines(i, 100 * r["q25"], 100 * r["q75"], color=COLORS["hb_ai"], lw=3, alpha=0.55)
            ax.plot(i, 100 * r["median"], "o", color=COLORS["hb_ai"], ms=8, mec=SURFACE, mew=1.5)
            ax.text(i + 0.12, 100 * r["median"], f"{100 * r['median']:.1f}", va="center", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(groups)), [f"{n}\n{g['n_pids']} people\n{g['n_nights']} nights" for n, g in groups.items()])
        ax.set_xlim(-0.5, len(groups) - 0.4)
        ax.axhline(overall, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 25)
    axes[0].set_ylabel("Disagreement, human vs headband (%)")
    fig.suptitle("Disagreement by age and sex: the groups overlap a lot", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.text(0.01, 0.005, "Dot = median over people, bar = middle 50 % of people (each person = mean of their nights).\n"
             f"Grey line = median over all people ({overall:.1f} %). Group sizes are small; no statistical test was run.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    agreement = json.loads(AGREEMENT.read_text())
    fig_confusion(agreement, FIG_DIR / "fig6_headband_confusion.png")
    fig_stage_f1(agreement, FIG_DIR / "fig7_per_stage_f1.png")
    fig_per_night_accuracy(agreement, FIG_DIR / "fig8_per_night_accuracy.png")
    fig_example_nights(load_all(DATA), agreement, FIG_DIR / "fig9_example_nights.png")
    hotspots = json.loads(HOTSPOTS.read_text())
    fig_dropout_by_night(hotspots, FIG_DIR / "fig10_headband_dropout_by_night.png")
    fig_groups(hotspots, FIG_DIR / "fig11_disagreement_by_age_sex.png")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
