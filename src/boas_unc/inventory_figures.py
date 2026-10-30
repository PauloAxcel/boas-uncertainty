"""Explanatory figures for docs/data_inventory.md (section 0).

Reads only the label files in data/boas (never the EDFs) and writes
figures/inventory/*.png plus results/inventory_counts.json.

Run from the repo root:  uv run python src/boas_unc/inventory_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from boas_unc.load import AGE_BANDS, load_all, runs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
FIG_DIR = ROOT / "figures" / "inventory"
RESULTS = ROOT / "results" / "inventory_counts.json"

# Codes as documented in data/boas/README ("Sleep staging labels").
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 8: "8 disc.", -2: "-2 art."}
SLEEP_STAGES = [0, 1, 2, 3, 4]
SOURCES = {"psg_hum": "Human (PSG)", "psg_ai": "AI on PSG", "hb_ai": "AI on headband"}

# Reference palette, light mode (dataviz skill references/palette.md).
COLORS = {"psg_hum": "#2a78d6", "psg_ai": "#eb6834", "hb_ai": "#1baf7a"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def pick_example(df: pd.DataFrame) -> str:
    """Night closest to median length among those showing all three special-code kinds."""
    g = df.groupby("sub")
    has_all = g.apply(lambda n: (n.psg_hum == 8).any() and (n.psg_ai == -2).any() and (n.hb_ai == -2).any())
    lengths = g.size()[has_all]
    return (lengths - lengths.median()).abs().sort_values(kind="stable").index[0]


# Hypnogram convention: Wake on top, REM next, then deepening NREM.
HYP_Y = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
HYP_LABELS = ["N3", "N2", "N1", "REM", "W"]


def fig_example_night(df: pd.DataFrame, sub: str, out: Path):
    night = df[df["sub"] == sub].sort_values("onset")
    hours = night.onset.to_numpy() / 3600
    step = 30 / 3600
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    for ax, (col, name) in zip(axes, SOURCES.items()):
        codes = night[col].to_numpy()
        y = np.array([HYP_Y.get(c, np.nan) for c in codes], dtype=float)
        ax.step(np.append(hours, hours[-1] + step), np.append(y, y[-1]), where="post",
                color=COLORS[col], lw=1.4)
        for code, label in [(8, "8 = PSG disconnected"), (-2, "-2 = artefact / missing")]:
            for start, length in runs(codes, code):
                # edge line keeps single-epoch runs (30 s, sub-pixel at this scale) visible
                ax.axvspan(hours[start], hours[start] + length * step, fc=MUTED, ec=MUTED,
                           alpha=0.35, lw=0.8)
        special = sorted(set(codes) & {8, -2})
        tag = "" if not special else "   grey bands: " + ", ".join(
            "8 = PSG disconnected" if c == 8 else "-2 = artefact / missing" for c in special)
        ax.set_title(f"{name}  ({col.replace('_', '.')}){tag}", color=INK, fontsize=10)
        ax.set_yticks(range(5), HYP_LABELS)
        ax.set_ylim(-0.5, 4.5)
        ax.grid(axis="y")
    axes[-1].set_xlabel("Hours since recording start (each step = one 30 s epoch)")
    fig.suptitle(f"One night, three label sources: {sub}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def stage_share(df: pd.DataFrame) -> pd.DataFrame:
    """Per night, percentage of epochs in each code, long format."""
    rows = []
    for col in SOURCES:
        pct = pd.crosstab(df["sub"], df[col], normalize="index") * 100
        for code in pct.columns:
            rows.append(pd.DataFrame({"sub": pct.index, "source": col, "code": code, "pct": pct[code].values}))
    return pd.concat(rows, ignore_index=True)


def fig_stage_share(share: pd.DataFrame, out: Path):
    codes = [0, 1, 2, 3, 4, 8, -2]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.26
    # centre the boxes of each code on its tick, however many sources use that code
    present = {code: [c for c in SOURCES if ((share.source == c) & (share.code == code)).any()] for code in codes}
    for col in SOURCES:
        pos, data = [], []
        for j, code in enumerate(codes):
            if col not in present[code]:
                continue
            vals = share[(share.source == col) & (share.code == code)].set_index("sub").pct
            # a night without the code has 0 % of it, not a missing value
            vals = vals.reindex(share["sub"].unique(), fill_value=0.0)
            k, n = present[code].index(col), len(present[code])
            pos.append(j + (k - (n - 1) / 2) * width)
            data.append(vals.values)
        bp = ax.boxplot(data, positions=pos, widths=width * 0.8, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED),
                        flierprops=dict(marker="o", ms=3, mfc=COLORS[col], mec="none", alpha=0.5))
        for box in bp["boxes"]:
            box.set(facecolor=COLORS[col], edgecolor=SURFACE, alpha=0.85)
        ax.plot([], [], "s", color=COLORS[col], label=SOURCES[col])
    short = {8: "disconnect", -2: "artefact"}
    ax.set_xticks(range(len(codes)), [f"{c}\n{short.get(c, STAGE_NAMES[c])}" for c in codes])
    ax.set_ylabel("% of all recorded epochs\n(not % of sleep)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("How each night divides into stages (one box = 128 nights; line = median)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_crosstab(df: pd.DataFrame, out: Path):
    order_r = [0, 1, 2, 3, 4, 8]
    order_c = [0, 1, 2, 3, 4, -2]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, col in zip(axes, ["psg_ai", "hb_ai"]):
        ct = pd.crosstab(df.psg_hum, df[col]).reindex(index=order_r, columns=order_c, fill_value=0)
        m = ct.to_numpy().astype(float)
        ax.imshow(np.where(m > 0, m, np.nan), cmap=cmap, norm=LogNorm(1, m.max()))
        for (r, c), v in np.ndenumerate(m):
            dark = v > 2000
            ax.text(c, r, f"{int(v):,}".replace(",", " "), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK)
        ax.set_xticks(range(len(order_c)), [STAGE_NAMES[c] for c in order_c])
        ax.set_yticks(range(len(order_r)), [STAGE_NAMES[c] for c in order_r])
        ax.set_xlabel(f"{SOURCES[col]} says…")
        ax.set_ylabel("Human (PSG) says…")
        ax.set_title(f"Human vs {SOURCES[col]}: epoch counts")
        for spine in ax.spines.values():
            spine.set_visible(False)
        # outline the special-code row/column
        ax.add_patch(plt.Rectangle((-0.5, 4.5), 6, 1, fill=False, ec=INK2, lw=1.2, ls="--"))
        ax.add_patch(plt.Rectangle((4.5, -0.5), 1, 6, fill=False, ec=INK2, lw=1.2, ls="--"))
    fig.suptitle("Every epoch counted once per panel (diagonal = same label; dashed = special codes)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_special_timeline(df: pd.DataFrame, out: Path):
    subs = df.groupby("sub").size().sort_values(kind="stable").index.tolist()  # shortest at bottom
    panels = [("psg_hum", 8, "Human: code 8 (PSG disconnected)"),
              ("psg_ai", -2, "AI on PSG: code -2"),
              ("hb_ai", -2, "AI on headband: code -2")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
    step = 30 / 3600
    for ax, (col, code, title) in zip(axes, panels):
        for y, sub in enumerate(subs):
            night = df[df["sub"] == sub].sort_values("onset")
            ax.plot([0, len(night) * step], [y, y], color=GRID, lw=1.6, solid_capstyle="butt")
            seq = night[col].to_numpy()
            hours = night.onset.to_numpy() / 3600
            for start, length in runs(seq, code):
                ax.plot([hours[start], hours[start] + length * step], [y, y],
                        color=COLORS[col], lw=2.4, solid_capstyle="butt")
        n_nights = df.groupby("sub")[col].apply(lambda s: (s == code).any()).sum()
        ax.set_title(f"{title}\n{n_nights} of {len(subs)} nights", fontsize=10)
        ax.set_xlabel("Hours since recording start")
        ax.set_yticks([])
    axes[0].set_ylabel("128 nights, sorted by length (grey = recorded, colour = code present)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def n3_pct_of_sleep(df: pd.DataFrame) -> pd.Series:
    """Per night, human N3 as % of human sleep epochs (N1-REM), the AASM denominator."""
    return df.groupby("sub").psg_hum.apply(lambda s: 100 * (s == 3).sum() / s.isin([1, 2, 3, 4]).sum())


def sleep_composition(df: pd.DataFrame) -> dict:
    n3 = n3_pct_of_sleep(df)
    pt = pd.read_csv(DATA / "participants.tsv", sep="\t", dtype=str, keep_default_na=False)
    age = pt.set_index("participant_id").age
    aged = n3[age.reindex(n3.index) != "n/a"]
    ages = age.reindex(aged.index).astype(float)
    bands = {}
    for lo, hi in AGE_BANDS:
        g = aged[(ages >= lo) & (ages <= hi)]
        bands[f"{lo}-{hi}"] = {"n_nights": int(len(g)), "median": round(float(g.median()), 2),
                               "mean": round(float(g.mean()), 2), "nights_zero_N3": int((g == 0).sum())}
    return {"human_N3_pct_of_sleep_per_night_q25_median_q75": [round(float(q), 2) for q in
                                                               np.percentile(n3, [25, 50, 75])],
            "human_N3_pct_of_sleep_by_age_band": bands,
            "human_N3_pct_of_sleep_vs_age_spearman": round(float(aged.corr(ages, method="spearman")), 3)}


def special_codes_vs_events(df: pd.DataFrame) -> dict:
    """Nights where headband -2 falls at/before human sleep onset or the first human REM epoch."""
    so_before, so_at, rem_before, rem_at = 0, [], 0, []
    hum8_before_onset, hum8_before_rem = 0, 0
    for sub, night in df.sort_values("onset").groupby("sub"):
        hum, hb = night.psg_hum.to_numpy(), night.hb_ai.to_numpy()
        onset = np.flatnonzero(np.isin(hum, [1, 2, 3, 4]))[0]
        so_before += bool((hb[:onset + 1] == -2).any())
        hum8_before_onset += bool((hum[:onset] == 8).any())
        if hb[onset] == -2:
            so_at.append(sub)
        rem = np.flatnonzero(hum == 4)
        if len(rem):
            rem_before += bool((hb[:rem[0] + 1] == -2).any())
            hum8_before_rem += bool((hum[:rem[0]] == 8).any())
            if hb[rem[0]] == -2:
                rem_at.append(sub)
    order = lambda subs: sorted(subs, key=lambda s: int(s[4:]))
    return {"nights_hb_minus2_at_or_before_human_sleep_onset": so_before,
            "nights_hb_minus2_on_human_sleep_onset_epoch": order(so_at),
            "nights_hb_minus2_at_or_before_first_human_REM": rem_before,
            "nights_hb_minus2_on_first_human_REM_epoch": order(rem_at),
            "nights_human_8_before_human_sleep_onset": hum8_before_onset,
            "nights_human_8_before_first_human_REM": hum8_before_rem}


def trailing_dropout_human_stages(df: pd.DataFrame) -> dict:
    """What the human scorers recorded during headband -2 runs that last until the end of the recording."""
    codes = []
    for _, night in df.sort_values("onset").groupby("sub"):
        hum, hb = night.psg_hum.to_numpy(), night.hb_ai.to_numpy()
        for start, length in runs(hb, -2):
            if start + length == len(hb):
                codes.extend(hum[start:])
    counts = pd.Series(codes).value_counts().sort_index()
    n_sleep = int(counts.reindex([1, 2, 3, 4], fill_value=0).sum())
    return {"n_epochs": int(len(codes)),
            "human_code_counts": {str(k): int(v) for k, v in counts.items()},
            "human_sleep_epochs": n_sleep,
            "pct_human_asleep_of_all_epochs": round(100 * n_sleep / len(codes), 2),
            "pct_human_asleep_of_epochs_with_stage_0_to_4": round(
                100 * n_sleep / int(counts.reindex(SLEEP_STAGES, fill_value=0).sum()), 2)}


def terminal_wake_minutes(df: pd.DataFrame) -> pd.Series:
    """Per night, minutes of human Wake (0) after the last human sleep epoch."""
    def one(seq: pd.Series) -> float:
        hum = seq.to_numpy()
        last = np.flatnonzero(np.isin(hum, [1, 2, 3, 4]))[-1]
        return 0.5 * int((hum[last + 1:] == 0).sum())
    return df.sort_values("onset").groupby("sub").psg_hum.apply(one)


def terminal_wake_summary(df: pd.DataFrame) -> dict:
    tw = terminal_wake_minutes(df)
    return {"human_terminal_wake_min_per_night_q25_median_q75_max": [
                *[round(float(q), 2) for q in np.percentile(tw, [25, 50, 75])], float(tw.max())],
            "nights_human_terminal_wake_over_5_min": int((tw > 5).sum())}


def summary(df: pd.DataFrame, share: pd.DataFrame, example: str) -> dict:
    out = {"n_nights": int(df["sub"].nunique()), "n_epochs": int(len(df)), "example_night": example}
    for col in SOURCES:
        counts = df[col].value_counts().sort_index()
        sleep = df[col].isin([1, 2, 3, 4])
        out[col] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "pct_of_sleep_epochs_pooled": {
                STAGE_NAMES[c]: round(100 * (df[col] == c).sum() / sleep.sum(), 2) for c in [1, 2, 3, 4]},
            "pct_of_night_per_night_median_iqr": {
                STAGE_NAMES[c]: [round(float(q), 2) for q in np.percentile(
                    share[(share.source == col) & (share.code == c)].set_index("sub").pct
                    .reindex(df["sub"].unique(), fill_value=0.0), [25, 50, 75])]
                for c in counts.index},
        }
    out["hb_ai_minus2_epochs_in_end_of_night_runs"] = int(sum(
        length for _, n in df.groupby("sub") for seq in [n.sort_values("onset").hb_ai.to_numpy()]
        for start, length in runs(seq, -2) if start + length == len(seq)))
    out["nights_without_human_N3"] = int((df.groupby("sub").psg_hum.apply(lambda s: (s == 3).sum()) == 0).sum())
    out.update(sleep_composition(df))
    out.update(special_codes_vs_events(df))
    out["hb_ai_minus2_end_of_night_runs_human_stages"] = trailing_dropout_human_stages(df)
    out.update(terminal_wake_summary(df))
    out["night_hours_min_median_max"] = [round(float(x) * 30 / 3600, 2) for x in
                                         np.percentile(df.groupby("sub").size(), [0, 50, 100])]
    return out


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all(DATA)
    share = stage_share(df)
    example = pick_example(df)
    fig_example_night(df, example, FIG_DIR / "fig1_example_night.png")
    fig_stage_share(share, FIG_DIR / "fig2_stage_share_per_night.png")
    fig_crosstab(df, FIG_DIR / "fig3_label_crosstab.png")
    fig_special_timeline(df, FIG_DIR / "fig4_special_codes_timeline.png")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(summary(df, share, example), indent=2))
    print(f"example night: {example}; wrote {FIG_DIR} and {RESULTS}")


if __name__ == "__main__":
    main()
