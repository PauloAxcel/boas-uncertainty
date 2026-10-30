"""Device-only abstention flags and agreement-coverage curves (Milestone 6). Definitions: docs/definitions.md, D14.

Flags use the headband label sequence alone (what a home user's device has). One reference flag,
`psg_ai_disagrees`, needs the PSG AI and is NOT available on the device: it is an upper bound.
Writes results/abstention.json and figures/abstention/*.png.

Run from the repo root:  uv run python -m boas_unc.abstain
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boas_unc import metrics as mt
from boas_unc.agreement import jsonable, kappa, summarise
from boas_unc.clinical import MATCHED_METRICS, STANDALONE_METRICS, all_nights, paired
from boas_unc.hotspots import both_stage, distance_to_nearest, transition_mask
from boas_unc.inventory_figures import GRID, INK, INK2, MUTED, SURFACE, style
from boas_unc.load import AI_MISSING, load_all

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
RESULTS = ROOT / "results" / "abstention.json"
AGREEMENT = ROOT / "results" / "agreement.json"
FIG_DIR = ROOT / "figures" / "abstention"

CHANGE_K = range(0, 6)
ARTEFACT_K = range(1, 6)
COMBINED_K = range(0, 6)
MIN_KEPT_EPOCHS = 10  # a night enters a per-night summary with at least this many kept epochs
PSG_FLAG = "psg_ai_disagrees"  # needs the PSG: not available on the device
CLINICAL_OPERATING_POINTS = ["isolated", "change_1", "artefact_2", "combined_1", PSG_FLAG]
CLINICAL_EXCLUSION_METRICS = [m for m in MATCHED_METRICS if m != "tst"]
CLINICAL_SMOOTHING_METRICS = STANDALONE_METRICS + ["rem_latency_sustained"]


def is_stage(codes: np.ndarray) -> np.ndarray:
    return np.isin(codes, (0, 1, 2, 3, 4))


def change_flag(hb: np.ndarray, k: int) -> np.ndarray:
    """Within k epochs of a predicted stage change (k = 0: the change epochs themselves)."""
    return distance_to_nearest(transition_mask(hb)) <= k


def isolated_flag(hb: np.ndarray) -> np.ndarray:
    """A single-epoch stage: both neighbours exist, are stages, and differ from this epoch."""
    stage = is_stage(hb)
    flag = np.zeros(len(hb), dtype=bool)
    if len(hb) < 3:
        return flag
    mid, before, after = hb[1:-1], hb[:-2], hb[2:]
    flag[1:-1] = stage[1:-1] & stage[:-2] & stage[2:] & (mid != before) & (mid != after)
    return flag


def artefact_flag(hb: np.ndarray, k: int) -> np.ndarray:
    """Within k epochs of a headband -2, not counting the -2 epochs themselves."""
    return (distance_to_nearest(hb == AI_MISSING) <= k) & (hb != AI_MISSING)


def device_flags(hb: np.ndarray) -> dict:
    """Every device-only flag set of one night, by name.

    An isolated epoch differs from both neighbours, so it is always a change epoch: `isolated`
    is a subset of `change_0`, and "isolated or change_k" would equal `change_k`.
    """
    flags = {"none": np.zeros(len(hb), dtype=bool), "isolated": isolated_flag(hb)}
    for k in CHANGE_K:
        flags[f"change_{k}"] = change_flag(hb, k)
    for k in ARTEFACT_K:
        flags[f"artefact_{k}"] = artefact_flag(hb, k)
    for k in COMBINED_K:
        flags[f"combined_{k}"] = flags[f"change_{k}"] | (artefact_flag(hb, k) if k else False)
    return flags


def psg_disagree_flag(psg_ai: np.ndarray, hb: np.ndarray) -> np.ndarray:
    """Reference only, not on the device: the PSG AI label differs from the headband label."""
    return psg_ai != hb


def smooth_isolated(hb: np.ndarray) -> np.ndarray:
    """Isolated epochs whose two neighbours carry the same label take that label."""
    out = hb.copy()
    fix = isolated_flag(hb)
    fix[1:-1] &= hb[:-2] == hb[2:]
    out[fix] = np.roll(hb, 1)[fix]  # the previous epoch's label, equal to the next one's
    return out


def family(name: str) -> str:
    """Flag family of a flag name, e.g. change_3 -> change."""
    return name.rsplit("_", 1)[0] if name[-1].isdigit() else name


def add_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column flag_<name> per flag set, plus the smoothed headband label hb_smooth."""
    parts = []
    for _, night in table.groupby("sub", sort=False):
        night = night.sort_values("epoch")
        hb = night.hb_ai.to_numpy()
        cols = {f"flag_{n}": f for n, f in device_flags(hb).items()}
        cols[f"flag_{PSG_FLAG}"] = psg_disagree_flag(night.psg_ai.to_numpy(), hb)
        cols["hb_smooth"] = smooth_isolated(hb)
        parts.append(night.assign(**cols))
    return pd.concat(parts, ignore_index=True)


def flag_names(df: pd.DataFrame) -> list[str]:
    return [c[len("flag_"):] for c in df.columns if c.startswith("flag_")]


def flag_summary(df: pd.DataFrame, flag: pd.Series, base: pd.Series, n_human: int) -> dict:
    """Pooled coverage and agreement of one flag set (human = reference, base = version (a) epochs)."""
    err = base & (df.psg_hum != df.hb_ai)
    kept, flagged = base & ~flag, base & flag
    n_base, n_kept, n_flagged = int(base.sum()), int(kept.sum()), int(flagged.sum())
    n_err = int(err.sum())
    return {"n_base": n_base, "n_kept": n_kept, "n_flagged": n_flagged,
            "coverage": n_kept / n_base, "coverage_of_human_epochs": n_kept / n_human,
            "accuracy_kept": 1 - int((err & ~flag).sum()) / n_kept if n_kept else np.nan,
            "kappa_kept": kappa(df.psg_hum.to_numpy()[kept.to_numpy()], df.hb_ai.to_numpy()[kept.to_numpy()])
            if n_kept else np.nan,
            "accuracy_flagged": 1 - int((err & flag).sum()) / n_flagged if n_flagged else np.nan,
            "error_capture": int((err & flag).sum()) / n_err if n_err else np.nan}


def per_night_flag(df: pd.DataFrame, flag: pd.Series, base: pd.Series) -> dict:
    """Per-night coverage and accuracy on kept epochs (median, IQR, nights defined)."""
    err = base & (df.psg_hum != df.hb_ai)
    g = pd.DataFrame({"sub": df["sub"], "base": base, "kept": base & ~flag, "err_kept": err & ~flag})
    n = g.groupby("sub", sort=False).sum()
    coverage = (n.kept / n.base).where(n.base > 0)
    accuracy = (1 - n.err_kept / n.kept).where(n.kept >= MIN_KEPT_EPOCHS)
    return {"coverage": summarise(coverage), "accuracy_kept": summarise(accuracy)}


def evaluate_flags(df: pd.DataFrame) -> dict:
    """Pooled and per-night results for every flag set."""
    base = both_stage(df)
    n_human = int(is_stage(df.psg_hum.to_numpy()).sum())
    out = {}
    for name in flag_names(df):
        flag = df[f"flag_{name}"]
        out[name] = {"family": family(name), "on_device": name != PSG_FLAG,
                     **flag_summary(df, flag, base, n_human), "per_night": per_night_flag(df, flag, base)}
    return out


def error_summary(human: pd.Series, headband: pd.Series) -> dict:
    """Bias, SD, width of the 95 % limits of agreement and median absolute error over paired nights."""
    h, b = paired(human, headband)
    d = b - h
    sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    return {"n": int(len(d)), "bias": float(d.mean()) if len(d) else np.nan, "sd_difference": sd,
            "loa_width": 2 * 1.96 * sd, "median_abs_error": float(d.abs().median()) if len(d) else np.nan}


def compare_errors(before: dict, after: dict) -> dict:
    """before, after and the change in |bias|, limits-of-agreement width and median absolute error."""
    return {"before": before, "after": after,
            "change_in_abs_bias": abs(after["bias"]) - abs(before["bias"]),
            "change_in_loa_width": after["loa_width"] - before["loa_width"],
            "change_in_median_abs_error": after["median_abs_error"] - before["median_abs_error"]}


def excluded_night_metrics(night: pd.DataFrame, flag: np.ndarray | None) -> dict:
    """Matched-epoch metrics of one night on the kept epochs (both stages, human 8 out, flagged out)."""
    hum, hb = night.psg_hum.to_numpy(), night.hb_ai.to_numpy()
    keep = mt.matched_mask(hum, hb, ~night.hum_disconnected.to_numpy())
    if flag is not None:
        keep = keep & ~flag
    return {**{f"hum_{k}": v for k, v in mt.matched_metrics(hum, keep).items()},
            **{f"hb_{k}": v for k, v in mt.matched_metrics(hb, keep).items()}}


def exclusion_effect(df: pd.DataFrame, names: list[str]) -> dict:
    """Clinical metric errors when flagged epochs are left out, against the unflagged matched analysis."""
    def metrics_by_night(name):
        rows = {}
        for sub, night in df.groupby("sub", sort=False):
            night = night.sort_values("epoch")
            rows[sub] = excluded_night_metrics(night, None if name is None else night[f"flag_{name}"].to_numpy())
        return pd.DataFrame.from_dict(rows, orient="index")
    base = metrics_by_night(None)
    out = {}
    for name in names:
        flagged = metrics_by_night(name)
        out[name] = {"on_device": name != PSG_FLAG,
                     "mean_tib_matched_min": {"before": float(base.hum_tib_matched.mean()),
                                              "after": float(flagged.hum_tib_matched.mean())},
                     "metrics": {m: compare_errors(error_summary(base["hum_" + m], base["hb_" + m]),
                                                   error_summary(flagged["hum_" + m], flagged["hb_" + m]))
                                 for m in CLINICAL_EXCLUSION_METRICS}}
    return out


def smoothing_effect(table: pd.DataFrame) -> dict:
    """Agreement and clinical metric errors before and after isolated-epoch smoothing."""
    smoothed = table.assign(hb_ai=table.hb_smooth)
    base = both_stage(table)
    changed = table.hb_ai != table.hb_smooth
    agree = {}
    for label, frame in (("before", table), ("after", smoothed)):
        h, b = frame.psg_hum.to_numpy()[base.to_numpy()], frame.hb_ai.to_numpy()[base.to_numpy()]
        agree[label] = {"n_epochs": int(len(h)), "accuracy": float((h == b).mean()), "kappa": kappa(h, b)}
    before, after = all_nights(table), all_nights(smoothed)
    return {"rule": "an isolated epoch whose two neighbours carry the same label takes that label",
            "n_epochs_changed": int(changed.sum()), "share_of_epochs_changed": float(changed.mean()),
            "agreement_version_a": agree,
            "clinical": {m: compare_errors(error_summary(before["hum_" + m], before["hb_" + m]),
                                           error_summary(after["hum_" + m], after["hb_" + m]))
                         for m in CLINICAL_SMOOTHING_METRICS}}


def run(table: pd.DataFrame, baseline_accuracy_from_agreement: float | None = None) -> dict:
    df = add_flags(table)
    flags = evaluate_flags(df)
    base = both_stage(df)
    base_acc = float(((df.psg_hum == df.hb_ai) & base).sum() / base.sum())
    return {
        "reference": "psg_hum", "candidate": "hb_ai", "base": "epochs where human and headband both give a stage 0-4",
        "baseline_accuracy": base_acc,
        "baseline_matches_agreement_json": None if baseline_accuracy_from_agreement is None
        else bool(abs(base_acc - baseline_accuracy_from_agreement) < 1e-6),  # the JSON is rounded to 6 decimals
        "random_abstention_expected_accuracy": base_acc,
        "flag_sets": flags,
        "clinical_excluded": exclusion_effect(df, CLINICAL_OPERATING_POINTS),
        "clinical_smoothed": smoothing_effect(df),
    }


CURVE_STYLE = {"change": ("#2a78d6", "o", "Within k epochs of a predicted change"),
               "artefact": ("#eb6834", "s", "Within k epochs of a headband -2"),
               "combined": ("#1baf7a", "^", "Near a predicted change or near a -2")}


def fig_curves(result: dict, out: Path):
    """Agreement on kept epochs against coverage, one line per flag family, k swept along each line."""
    flags = result["flag_sets"]
    base = 100 * result["baseline_accuracy"]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.axhline(base, color=MUTED, lw=1.0)
    ax.text(0.30, base + 0.25, f"chance: random abstention leaves it at {base:.1f} %", fontsize=8, color=MUTED,
            transform=ax.get_yaxis_transform())
    for fam, (color, marker, label) in CURVE_STYLE.items():
        pts = sorted((f["coverage"], f["accuracy_kept"], n) for n, f in flags.items() if f["family"] == fam)
        pts.append((flags["none"]["coverage"], flags["none"]["accuracy_kept"], "none"))  # every curve starts here
        x, y = [100 * p[0] for p in pts], [100 * p[1] for p in pts]
        ax.plot(x, y, color=color, lw=1.6, marker=marker, ms=6, mec=SURFACE, mew=1.1, label=label)
        if fam == "change":
            for xi, yi, p in zip(x, y, pts[:-1]):
                ax.annotate(f"k={p[2].rsplit('_', 1)[1]}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color=INK2)
    iso = flags["isolated"]
    ax.plot(100 * iso["coverage"], 100 * iso["accuracy_kept"], marker="v", ms=8, color="#e87ba4", mec=SURFACE, mew=1.1,
            ls="none", label="Isolated single-epoch stage")
    psg = flags[PSG_FLAG]
    ax.plot(100 * psg["coverage"], 100 * psg["accuracy_kept"], marker="D", ms=9, mfc="none", mec=INK, mew=1.6,
            ls="none", label="PSG AI disagrees: NOT available on the device")
    ax.set_xlabel("Coverage: share of epochs the device keeps (%)")
    ax.set_ylabel("Agreement with the human on kept epochs (%)")
    ax.invert_xaxis()
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=INK2)
    ax.set_title("Dropping suspicious epochs raises agreement, at a price in coverage")
    fig.text(0.01, 0.005, "Version (a) epochs. Every device-only line starts at 100 % coverage with the Milestone 3 "
             "agreement (left). Values: results/abstention.json", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    expected = json.loads(AGREEMENT.read_text())["pairs"]["human_vs_hb_ai"]["a_both_stage"]["pooled"]["accuracy"]
    result = jsonable(run(load_all(DATA), expected))
    RESULTS.write_text(json.dumps(result, indent=1))
    fig_curves(result, FIG_DIR / "fig14_agreement_vs_coverage.png")
    print(f"baseline {100 * result['baseline_accuracy']:.2f} % | matches agreement.json: "
          f"{result['baseline_matches_agreement_json']} | wrote {RESULTS} and {FIG_DIR}")


if __name__ == "__main__":
    main()
