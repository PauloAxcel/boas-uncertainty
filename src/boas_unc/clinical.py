"""Clinical sleep metrics per night, human vs headband, with Bland-Altman statistics (Milestone 5).

Definitions: docs/definitions.md (D2-D6, D8, D9, D11). The single-sequence metrics are in
metrics.py. Differences are headband minus human. Writes results/clinical.json and
figures/clinical/*.png.

Run from the repo root:  uv run python -m boas_unc.clinical
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from boas_unc import metrics as mt
from boas_unc.agreement import jsonable, summarise
from boas_unc.inventory_figures import COLORS, GRID, INK, INK2, MUTED, SURFACE, style
from boas_unc.load import AI_MISSING, HUM_DISCONNECTED, load_all, read_participants

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
RESULTS = ROOT / "results" / "clinical.json"
FIG_DIR = ROOT / "figures" / "clinical"

STANDALONE_METRICS = ["tst", "sleep_efficiency", "sol", "waso", "terminal_wake", "rem_latency",
                      "pre_sleep_wake", "pct_n1", "pct_n2", "pct_n3", "pct_rem"]
MATCHED_METRICS = ["tst", "sleep_efficiency", "pct_n1", "pct_n2", "pct_n3", "pct_rem"]
DROPOUT_THRESHOLDS = {"20_percent_primary": 0.20, "10_percent": 0.10, "all_nights": None}  # fixed before the results
LABELS = {"tst": ("Total sleep time", "min"), "sleep_efficiency": ("Sleep efficiency", "%"),
          "sol": ("Sleep onset latency", "min"), "waso": ("Wake after sleep onset (WASO)", "min"),
          "terminal_wake": ("Terminal wake", "min"), "rem_latency": ("REM latency", "min"),
          "pre_sleep_wake": ("Pre-sleep wake", "min"), "pct_n1": ("N1, share of sleep", "%"),
          "pct_n2": ("N2, share of sleep", "%"), "pct_n3": ("N3, share of sleep", "%"),
          "pct_rem": ("REM, share of sleep", "%")}
MAIN_FIGURE_METRICS = ["sleep_efficiency", "waso"]


def night_row(night: pd.DataFrame, treat_8_as_wake: bool = False) -> dict:
    """Every metric of one night, for the human (`hum_`) and the headband (`hb_`) sequence.

    Standalone metrics sit under hum_/hb_, matched-epoch metrics under mhum_/mhb_.
    With treat_8_as_wake (sensitivity check, D2) human 8 becomes Wake and nothing is excluded.
    """
    onset = night.onset.to_numpy()
    hum, hb = night.psg_hum.to_numpy(), night.hb_ai.to_numpy()
    if treat_8_as_wake:
        hum = np.where(hum == HUM_DISCONNECTED, mt.WAKE, hum)
        valid = np.ones(len(hum), dtype=bool)
    else:
        valid = ~night.hum_disconnected.to_numpy()
    matched = mt.matched_mask(hum, hb, valid)
    row = {}
    for prefix, seq in (("hum_", hum), ("hb_", hb)):
        row.update({prefix + k: v for k, v in mt.sleep_metrics(seq, onset, valid).items()})
        row.update({"m" + prefix + k: v for k, v in mt.matched_metrics(seq, matched).items()})
    row["hb_dropout_share"] = row["hb_unscored"] / row["hb_tib"]
    i, r = mt.first_sleep_index(hum, valid), np.flatnonzero(valid & (hum == mt.REM))
    row["flag_sol_dropout"] = bool(i is not None and (hb[:i + 1] == AI_MISSING).any())
    row["flag_rem_dropout"] = bool(len(r) and (hb[:r[0] + 1] == AI_MISSING).any())
    return row


def all_nights(table: pd.DataFrame, treat_8_as_wake: bool = False) -> pd.DataFrame:
    """night_row for every night, one row per night."""
    rows = {sub: night_row(n.sort_values("epoch"), treat_8_as_wake) for sub, n in table.groupby("sub", sort=False)}
    return pd.DataFrame.from_dict(rows, orient="index")


def ba_stats(human: np.ndarray, headband: np.ndarray) -> dict:
    """Bland-Altman statistics of paired values; differences are headband - human."""
    n = len(human)
    d = headband - human
    sd = float(np.std(d, ddof=1)) if n > 1 else np.nan
    bias = float(d.mean()) if n else np.nan
    can_correlate = n >= 3 and np.ptp(human) > 0 and np.ptp(headband) > 0
    return {"n": int(n), "mean_human": float(human.mean()) if n else np.nan,
            "mean_headband": float(headband.mean()) if n else np.nan,
            "bias": bias, "sd_difference": sd, "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd,
            "pearson_r": float(stats.pearsonr(human, headband)[0]) if can_correlate else np.nan,
            "spearman_rho": float(stats.spearmanr(human, headband)[0]) if can_correlate else np.nan}


def paired(human: pd.Series, headband: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The nights where both values are defined."""
    ok = human.notna() & headband.notna()
    return human[ok], headband[ok]


def bland_altman(human: pd.Series, headband: pd.Series, pid_of: pd.Series | None = None) -> dict:
    """Per-night Bland-Altman statistics, with the nights used and excluded and why (D5).

    With pid_of, the nights of each person are averaged first (D7) and n counts people.
    """
    h, b = paired(human, headband)
    out = {"n_nights_used": int(len(h)),
           "excluded": {"human_undefined": int((human.isna() & headband.notna()).sum()),
                        "headband_undefined": int((human.notna() & headband.isna()).sum()),
                        "both_undefined": int((human.isna() & headband.isna()).sum())},
           "human": summarise(human), "headband": summarise(headband), "difference": summarise(b - h),
           **ba_stats(h.to_numpy(), b.to_numpy())}
    if pid_of is not None:
        pid = pid_of.reindex(h.index)
        out["by_person"] = ba_stats(h.groupby(pid).mean().to_numpy(), b.groupby(pid).mean().to_numpy())
    return out


def metric_block(nights: pd.DataFrame, metrics: list[str], pid_of: pd.Series, hum="hum_", hb="hb_") -> dict:
    return {m: bland_altman(nights[hum + m], nights[hb + m], pid_of) for m in metrics}


def zero_floor_counts(nights: pd.DataFrame, metric: str) -> dict:
    """How many nights sit at exactly zero in one or both sequences (they fall on lines in a Bland-Altman plot)."""
    h, b = paired(nights["hum_" + metric], nights["hb_" + metric])
    return {"n": int(len(h)), "both_zero": int(((h == 0) & (b == 0)).sum()),
            "headband_zero_human_above": int(((b == 0) & (h > 0)).sum()),
            "human_zero_headband_above": int(((h == 0) & (b > 0)).sum())}


def identity_error(nights: pd.DataFrame) -> float:
    """Largest |TIB - (pre-sleep wake + TST + WASO + terminal wake + unscored)| over nights and sequences."""
    err = 0.0
    for p in ("hum_", "hb_"):
        total = sum(nights[p + k] for k in ("pre_sleep_wake", "tst", "waso", "terminal_wake", "unscored"))
        err = max(err, float((nights[p + "tib"] - total).abs().max()))
    return err


def run(table: pd.DataFrame, participants: pd.DataFrame) -> dict:
    nights = all_nights(table)
    pid_of = participants.set_index("participant_id").pid
    matched = {}
    for name, thr in DROPOUT_THRESHOLDS.items():
        keep = nights.index if thr is None else nights.index[nights.hb_dropout_share <= thr]
        matched[name] = {"threshold": thr, "n_nights": int(len(keep)),
                         "excluded_nights": [s for s in nights.index if s not in set(keep)],
                         "metrics": metric_block(nights.loc[keep], MATCHED_METRICS, pid_of, "mhum_", "mhb_")}
    nights8 = all_nights(table, treat_8_as_wake=True)
    no_sol_flag, no_rem_flag = nights[~nights.flag_sol_dropout], nights[~nights.flag_rem_dropout]
    return {
        "n_nights": int(len(nights)), "difference": "headband minus human",
        "identity_max_abs_error_min": identity_error(nights),
        "standalone": metric_block(nights, STANDALONE_METRICS, pid_of),
        "zero_floor": {m: zero_floor_counts(nights, m) for m in STANDALONE_METRICS},
        "matched_epoch": matched,
        "sensitivity": {
            "human_8_as_wake": metric_block(nights8, STANDALONE_METRICS, pid_of),
            "latency_without_dropout_flagged_nights": {
                "sol": {"n_nights_flagged": int(nights.flag_sol_dropout.sum()),
                        **bland_altman(no_sol_flag.hum_sol, no_sol_flag.hb_sol, pid_of)},
                "rem_latency": {"n_nights_flagged": int(nights.flag_rem_dropout.sum()),
                                **bland_altman(no_rem_flag.hum_rem_latency, no_rem_flag.hb_rem_latency, pid_of)}},
            "rem_latency_sustained_run_of_3": bland_altman(
                nights.hum_rem_latency_sustained, nights.hb_rem_latency_sustained, pid_of),
            "waso_including_terminal_wake": bland_altman(nights.hum_waso_incl_terminal, nights.hb_waso_incl_terminal, pid_of),
        },
        "flags": {"left_censored_human": nights.index[nights.hum_left_censored].tolist(),
                  "left_censored_headband": nights.index[nights.hb_left_censored].tolist(),
                  "headband_dropout_over_20_percent": nights.index[nights.hb_dropout_share > 0.20].tolist(),
                  "sol_dropout_flagged": nights.index[nights.flag_sol_dropout].tolist(),
                  "rem_latency_dropout_flagged": nights.index[nights.flag_rem_dropout].tolist()},
        "per_night_metrics": nights.to_dict(orient="index"),
    }


def draw_ba(ax, result: dict, metric: str, compact: bool = False):
    """One Bland-Altman panel: difference (headband - human) against the mean of the two."""
    nights = pd.DataFrame(result["per_night_metrics"]).T
    title, unit = LABELS[metric]
    h, b = paired(nights["hum_" + metric].astype(float), nights["hb_" + metric].astype(float))
    mean, diff = (h + b) / 2, b - h
    high = nights.loc[h.index, "hb_dropout_share"].astype(float) > 0.20
    ax.scatter(mean[~high], diff[~high], s=14 if compact else 22, color=COLORS["hb_ai"], alpha=0.65, edgecolors="none",
               label="night")
    ax.scatter(mean[high], diff[high], s=26 if compact else 38, facecolors="none", edgecolors=INK, linewidths=1.1,
               label="night with more than 20 % headband dropout")
    st = result["standalone"][metric]
    for y, text, lw, col in [(st["bias"], f"bias {st['bias']:+.1f}", 1.4, INK),
                             (st["loa_upper"], f"+1.96 SD {st['loa_upper']:+.1f}", 0.9, MUTED),
                             (st["loa_lower"], f"−1.96 SD {st['loa_lower']:+.1f}", 0.9, MUTED)]:
        ax.axhline(y, color=col, lw=lw)
        ax.text(1.0, y, text, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7 if compact else 8,
                color=INK2)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_title(f"{title} ({unit}), n = {st['n']} nights", fontsize=9 if compact else 10)
    ax.set_xlabel(f"Mean of human and headband ({unit})", fontsize=8 if compact else 9)
    ax.set_ylabel(f"Headband − human ({unit})", fontsize=8 if compact else 9)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def fig_bland_altman(result: dict, metrics: list[str], out: Path):
    """The main figure: Bland-Altman plots for the two chosen metrics."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 5.0))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        draw_ba(ax, result, metric)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2, fontsize=8, labelcolor=INK2,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Headband vs human: how far apart are the sleep metrics, night by night?", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.text(0.01, 0.005, "Each dot is one night. Lines: mean difference (bias) and 95 % limits of agreement "
             "(bias ± 1.96 SD). Standalone analysis: headband dropout counts as lost time.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_bland_altman_all(result: dict, out: Path):
    """Supplement: the same plot for every metric."""
    metrics = STANDALONE_METRICS
    fig, axes = plt.subplots(3, 4, figsize=(17, 11.5))
    for ax, metric in zip(axes.ravel(), metrics):
        draw_ba(ax, result, metric, compact=True)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    axes.ravel()[len(metrics)].legend(*axes.ravel()[0].get_legend_handles_labels(), frameon=False, loc="center",
                                      fontsize=9, labelcolor=INK2)
    fig.suptitle("Bland-Altman plots for every clinical metric (headband − human)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result = jsonable(run(load_all(DATA), read_participants(DATA)))
    RESULTS.write_text(json.dumps(result, indent=1))
    fig_bland_altman(result, MAIN_FIGURE_METRICS, FIG_DIR / "fig12_bland_altman.png")
    fig_bland_altman_all(result, FIG_DIR / "fig13_bland_altman_all_metrics.png")
    print(f"wrote {RESULTS} and {FIG_DIR}")


if __name__ == "__main__":
    main()
