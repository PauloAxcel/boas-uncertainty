"""What does the headband signal look like for each human stage? (D19)

Definitions: docs/definitions.md, "What the headband signal looks like for each human stage". Reads only
complete headband EDF files (size and SHA256 checked), read-only, and the label files. Checks consistency
between the headband signal and the PSG-based human labels. It cannot show that the human labels are right.
Writes results/stage_signature.json and figures/signals/fig17_stage_signature.png.

Run from the repo root:  uv run python -m boas_unc.stage_signature
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats
from sklearn.metrics import roc_auc_score

from matplotlib.colors import LinearSegmentedColormap

from boas_unc import edf
from boas_unc import signal_quality as sq
from boas_unc.agreement import jsonable, summarise
from boas_unc.inventory_figures import GRID, INK, MUTED, SEQ_BLUE, style
from boas_unc.load import STAGE_LABELS, load_night

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "stage_signature.json"
FIG = ROOT / "figures" / "signals" / "fig17_stage_signature.png"

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
TOTAL_BAND = (0.5, 30.0)
COMBINED = "combined"
VIEWS = list(sq.CHANNELS) + [COMBINED]
CONTRASTS = {"N3_vs_N2": ((3,), (2,)), "N2_vs_N1": ((2,), (1,)), "REM_vs_N2": ((4,), (2,)),
             "W_vs_sleep": ((0,), (1, 2, 3, 4))}
MIN_STAGE_EPOCHS = 10
MIN_CONTRAST_EPOCHS = 20
SPIKE_FACTOR = 3.0


def relative_band_powers(freqs: np.ndarray, psd: np.ndarray) -> pd.DataFrame:
    """Relative power of each band (band power over the 0.5-30 Hz total); NaN where the total power is zero."""
    df = freqs[1] - freqs[0]
    in_total = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    total = psd[:, in_total].sum(axis=1) * df
    out = {}
    for name, (lo, hi) in BANDS.items():
        keep = (freqs >= lo) & (freqs < hi) if hi < TOTAL_BAND[1] else (freqs >= lo) & (freqs <= hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = np.where(total > 0, psd[:, keep].sum(axis=1) * df / total, np.nan)
    return pd.DataFrame(out)


def channel_table(path: Path, header: dict, channel: str) -> pd.DataFrame:
    """Per epoch: relative band powers and log10 of the standard deviation for one channel."""
    sig = header["signals"][edf.signal_index(header, channel)]
    x = edf.read_signal(path, channel, header)
    ep = edf.epochs(x, sig["sampling_hz"], sq.EPOCH_S)
    freqs, psd = sps.welch(ep, fs=sig["sampling_hz"], nperseg=int(sq.WELCH_S * sig["sampling_hz"]), noverlap=0, axis=1)
    constant = sq.constant_epochs(ep)  # a constant epoch has no power, only rounding noise: undefined
    psd[constant] = 0.0
    out = relative_band_powers(freqs, psd)
    std = np.where(constant, 0.0, ep.std(axis=1))
    out["log_std"] = np.where(std > 0, np.log10(np.where(std > 0, std, 1.0)), np.nan)
    return out


def combine(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean over channels, ignoring a channel that is undefined for the epoch (D19)."""
    return pd.concat(tables).groupby(level=0).mean()


def night_views(path: Path, header: dict) -> dict[str, pd.DataFrame]:
    """The per-epoch tables of each channel and of the combined channel."""
    tables = {ch: channel_table(path, header, ch) for ch in sq.CHANNELS}
    return {**tables, COMBINED: combine(list(tables.values()))}


def stage_medians(view: pd.DataFrame, stage: np.ndarray) -> dict:
    """Median relative power per band for each human stage with enough epochs."""
    out = {}
    for code, name in STAGE_LABELS.items():
        rows = view[(stage == code) & view.delta.notna().to_numpy()]
        if len(rows) >= MIN_STAGE_EPOCHS:
            out[name] = {**{b: float(rows[b].median()) for b in BANDS}, "n_epochs": int(len(rows))}
    return out


def contrast_auroc(values: np.ndarray, stage: np.ndarray, first: tuple, second: tuple) -> float:
    """AUROC for telling the first group of stages from the second (above 0.5: higher in the first). NaN if too few."""
    ok = ~np.isnan(values)
    a, b = ok & np.isin(stage, first), ok & np.isin(stage, second)
    if a.sum() < MIN_CONTRAST_EPOCHS or b.sum() < MIN_CONTRAST_EPOCHS:
        return np.nan
    y = np.r_[np.ones(a.sum()), np.zeros(b.sum())]
    return float(roc_auc_score(y, np.r_[values[a], values[b]]))


def night_aurocs(view: pd.DataFrame, stage: np.ndarray) -> dict:
    """AUROC per contrast and per feature (each band and log std) within one night."""
    return {c: {f: contrast_auroc(view[f].to_numpy(), stage, *groups) for f in list(BANDS) + ["log_std"]}
            for c, groups in CONTRASTS.items()}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def spike_epochs(std_uv: np.ndarray, factor: float = SPIKE_FACTOR) -> np.ndarray:
    """Epochs whose amplitude is at least `factor` times the median over live (non-zero) epochs."""
    live = std_uv > 0
    if not live.any():
        return np.zeros(len(std_uv), dtype=bool)
    return live & (std_uv >= factor * np.median(std_uv[live]))


def night_result(views: dict[str, pd.DataFrame], stage: np.ndarray, std_hb1: np.ndarray, movement) -> dict:
    """Everything D19 asks for one night."""
    usable = np.isin(stage, list(STAGE_LABELS)) & views[COMBINED].delta.notna().to_numpy()
    n = min(len(stage), len(usable))
    spikes = spike_epochs(std_hb1[:n])
    h1, h2 = views[sq.CHANNELS[0]][:n], views[sq.CHANNELS[1]][:n]
    out = {"n_epochs_used": int(usable[:n].sum()),
           "stage_medians": {v: stage_medians(views[v][:n], np.where(usable[:n], stage[:n], -9)) for v in VIEWS},
           "aurocs": {v: night_aurocs(views[v][:n], np.where(usable[:n], stage[:n], -9)) for v in VIEWS},
           "channel_corr_log_std": spearman(h1.log_std.to_numpy(), h2.log_std.to_numpy()),
           "channel_corr_delta": spearman(h1.delta.to_numpy(), h2.delta.to_numpy()),
           "spike_epochs": int((spikes & usable[:n]).sum()),
           "spikes_by_stage": {name: int((spikes & usable[:n] & (stage[:n] == code)).sum()) for code, name in STAGE_LABELS.items()},
           "live_by_stage": {name: int((usable[:n] & (stage[:n] == code)).sum()) for code, name in STAGE_LABELS.items()}}
    if movement is not None:
        m = min(n, len(movement))
        out["spearman_log_std_vs_movement"] = spearman(h1.log_std.to_numpy()[:m],
                                                       np.log10(np.maximum(movement[:m], 1e-9)))
    return out


def aggregate(per_night: dict) -> dict:
    """Median and IQR over nights of the stage signatures, AUROCs and channel correlations, and pooled spike shares."""
    out = {"n_nights": len(per_night), "stage_signature": {}, "auroc": {}}
    for v in VIEWS:
        sig = {}
        for stage in STAGE_LABELS.values():
            rows = [r["stage_medians"][v][stage] for r in per_night.values() if stage in r["stage_medians"][v]]
            sig[stage] = {"n_nights": len(rows), **{b: summarise(pd.Series([r[b] for r in rows])) for b in BANDS}}
        out["stage_signature"][v] = sig
        out["auroc"][v] = {c: {f: {**summarise(pd.Series([r["aurocs"][v][c][f] for r in per_night.values()]))}
                               for f in list(BANDS) + ["log_std"]} for c in CONTRASTS}
    out["channel_corr_log_std"] = summarise(pd.Series([r["channel_corr_log_std"] for r in per_night.values()]))
    out["channel_corr_delta"] = summarise(pd.Series([r["channel_corr_delta"] for r in per_night.values()]))
    spikes = {s: sum(r["spikes_by_stage"][s] for r in per_night.values()) for s in STAGE_LABELS.values()}
    live = {s: sum(r["live_by_stage"][s] for r in per_night.values()) for s in STAGE_LABELS.values()}
    out["spikes"] = {"n_spike_epochs": sum(spikes.values()), "n_live_epochs": sum(live.values()),
                     "share_of_spikes_by_stage": {s: spikes[s] / max(sum(spikes.values()), 1) for s in spikes},
                     "share_of_live_epochs_by_stage": {s: live[s] / max(sum(live.values()), 1) for s in live}}
    mv = [r["spearman_log_std_vs_movement"] for r in per_night.values() if "spearman_log_std_vs_movement" in r]
    out["spearman_log_std_vs_movement"] = summarise(pd.Series(mv))
    return out


def run(data: Path, edf_dir: Path, subs: list[str] | None = None) -> dict:
    ok, skipped = sq.complete_nights(data, edf_dir, subs)
    per_night = {}
    for sub in ok:
        path = sq.edf_path(edf_dir, sub)
        header = edf.read_header(path)
        night = load_night(data, sub).sort_values("epoch").reset_index(drop=True)
        views = night_views(path, header)
        n = min(len(night), len(views[COMBINED]))
        std_hb1 = sq.night_quality(path, header)[f"{sq.CHANNELS[0]}_std_uv"].to_numpy()
        per_night[sub] = night_result({v: t.iloc[:n] for v, t in views.items()}, night.psg_hum.to_numpy()[:n],
                                      std_hb1[:n], sq.movement_per_epoch(path, header))
    return {"data_rule": "complete files only, read-only; checks consistency with the human labels, not their correctness",
            "nights_used": ok, "n_nights_skipped": len(skipped), **aggregate(per_night), "per_night": per_night}


def fig_signature(result: dict, out: Path):
    """Left: share of EEG power in each band, by human stage (combined channel). Right: N3 against N2, per night."""
    sig = result["stage_signature"][COMBINED]
    stages = [s for s in STAGE_LABELS.values() if sig[s]["n_nights"] > 0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    values = np.array([[100 * sig[s][band]["median"] for band in BANDS] for s in stages])
    span = values.max(axis=0) - values.min(axis=0)
    shade = (values - values.min(axis=0)) / np.where(span > 0, span, 1.0)  # each band shaded across stages
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    a.imshow(shade, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(values):
        a.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=10, color="#ffffff" if shade[r, c] > 0.55 else INK)
    a.set_xticks(range(len(BANDS)), [f"{k}\n{lo:g}-{hi:g} Hz" for k, (lo, hi) in BANDS.items()])
    a.set_yticks(range(len(stages)), [f"{s}  ({sig[s]['n_nights']} nights)" for s in stages])
    a.xaxis.tick_top()
    a.tick_params(length=0)
    for spine in a.spines.values():
        spine.set_visible(False)
    a.set_title("Share of EEG power (%, 0.5-30 Hz) in each band, by human stage\n"
                "median over nights, channels combined; darker = higher than the other stages", loc="left", fontsize=9.5,
                pad=34)
    per = result["per_night"]
    for i, v in enumerate(VIEWS):
        vals = np.array([r["aurocs"][v]["N3_vs_N2"]["delta"] for r in per.values()], dtype=float)
        vals = vals[~np.isnan(vals)]
        b.scatter(i + np.random.default_rng(0).uniform(-0.18, 0.18, len(vals)), vals, s=16, color="#2a78d6", alpha=0.6,
                  edgecolors="none")
        if len(vals):
            b.hlines(np.median(vals), i - 0.3, i + 0.3, color=INK, lw=2)
    b.axhline(0.5, color=MUTED, lw=0.8)
    b.set_xticks(range(len(VIEWS)), ["HB_1 alone", "HB_2 alone", "channels\ncombined"])
    b.set_ylabel("AUROC of delta power, N3 against N2\n(0.5 = no better than chance)")
    b.set_ylim(0.3, 1.02)
    b.grid(axis="y", color=GRID, lw=0.5)
    b.set_axisbelow(True)
    b.set_title("Does delta power tell N3 from N2? One dot per night", loc="left", fontsize=9.5, pad=34)
    fig.text(0.01, 0.005, "Consistency of the headband signal with the human labels (which come from the PSG). Values: "
             "results/stage_signature.json", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    style()
    result = jsonable(run(sq.DATA, sq.EDF_DIR))
    RESULTS.write_text(json.dumps(result, indent=1))
    fig_signature(result, FIG)
    print(f"{result['n_nights']} nights; wrote {RESULTS} and {FIG}")


if __name__ == "__main__":
    main()
