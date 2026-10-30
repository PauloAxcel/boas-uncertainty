"""Learned abstention from the headband labels alone (D16). Definitions: docs/definitions.md.

A small model predicts, from the headband label sequence and the time since the recording start only,
where the headband disagrees with the human. No signal is read, and no human or PSG AI label is a feature.
Every score is out of fold, with folds that keep all nights of one person together.
Writes results/learned_abstention.json and figures/abstention/fig16_learned_vs_heuristic.png.

Run from the repo root (after boas_unc.abstain):  uv run python -m boas_unc.learned_abstain
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from boas_unc import abstain as ab
from boas_unc.agreement import jsonable, summarise
from boas_unc.hotspots import both_stage, distance_from_prev, distance_to_next
from boas_unc.inventory_figures import GRID, INK, INK2, MUTED, SURFACE, style
from boas_unc.load import AI_MISSING, load_all, read_participants

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
ABSTENTION = ROOT / "results" / "abstention.json"
RESULTS = ROOT / "results" / "learned_abstention.json"
FIG_DIR = ROOT / "figures" / "abstention"

CATEGORIES = [0, 1, 2, 3, 4, AI_MISSING]
EDGE = 9  # label position before the first or after the last epoch
CAUSAL_LAGS = (0, -1, -2, -3)
NONCAUSAL_LAGS = (-3, -2, -1, 0, 1, 2, 3)
RUN_CAP = 40
FLAG_CAP = 20
N_SPLITS = 5
MIN_KEPT_EPOCHS = 10
COVERAGE_GRID = np.round(np.arange(1.0, 0.49, -0.02), 2)
MATCHED_NIGHT_FLAG = "change_1"  # per-night summary at the coverage of this hand-made flag
CONTEXT_FLAGS = ["isolated", "change_0", "change_1", "change_3", "artefact_2", "psg_ai_disagrees"]

MODELS = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}


def lagged(hb: np.ndarray, lag: int) -> np.ndarray:
    """The label `lag` epochs away (negative = earlier); EDGE where that epoch does not exist."""
    out = np.full(len(hb), EDGE)
    if lag == 0:
        return hb.copy()
    if lag < 0:
        out[-lag:] = hb[:lag]
    else:
        out[:-lag] = hb[lag:]
    return out


def one_hot(codes: np.ndarray) -> np.ndarray:
    return np.stack([codes == c for c in CATEGORIES + [EDGE]], axis=1).astype(float)


def change_epochs(hb: np.ndarray) -> np.ndarray:
    """True where a new stage begins: this epoch and the one before are stages 0-4 and differ."""
    stage = ab.is_stage(hb)
    out = np.zeros(len(hb), dtype=bool)
    out[1:] = stage[1:] & stage[:-1] & (hb[1:] != hb[:-1])
    return out


def capped(d: np.ndarray, cap: int) -> np.ndarray:
    return np.minimum(np.where(np.isinf(d), cap, d), cap).astype(float)


def window_share(x: np.ndarray, back: int, forward: int, as_count: bool = False) -> np.ndarray:
    """Share (or count) of True in the window from `back` epochs before to `forward` epochs after each epoch."""
    n = len(x)
    c = np.concatenate([[0], np.cumsum(x.astype(float))])
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - back), np.minimum(n - 1, idx + forward)
    total = c[hi + 1] - c[lo]
    return total if as_count else total / (hi - lo + 1)


def night_features(hb: np.ndarray, onset: np.ndarray, causal: bool) -> np.ndarray:
    """Features of every epoch of one night from the headband labels and the clock only (D16)."""
    lags = CAUSAL_LAGS if causal else NONCAUSAL_LAGS
    change, missing = change_epochs(hb), hb == AI_MISSING
    cols = [one_hot(lagged(hb, lag)) for lag in lags]
    cols.append(capped(distance_from_prev(change), RUN_CAP)[:, None])
    cols.append(capped(distance_from_prev(missing), FLAG_CAP)[:, None])
    if causal:
        cols.append(window_share(change, 10, 0, as_count=True)[:, None])
        cols.append(window_share(missing, 20, 0)[:, None])
    else:
        cols.append(capped(distance_to_next(change), RUN_CAP)[:, None])
        cols.append(capped(distance_to_next(missing), FLAG_CAP)[:, None])
        cols.append(window_share(change, 5, 5, as_count=True)[:, None])
        cols.append(window_share(missing, 10, 10)[:, None])
    cols.append((onset / 3600)[:, None])
    return np.hstack(cols)


def build_dataset(table: pd.DataFrame) -> dict:
    """Features for both sets, and the epoch-level target, for every epoch of every night."""
    parts, meta = {True: [], False: []}, []
    for sub, night in table.groupby("sub", sort=False):
        night = night.sort_values("epoch")
        hb, onset = night.hb_ai.to_numpy(), night.onset.to_numpy()
        for causal in (True, False):
            parts[causal].append(night_features(hb, onset, causal))
        base = ab.is_stage(night.psg_hum.to_numpy()) & ab.is_stage(hb)
        meta.append(pd.DataFrame({"sub": sub, "pid": night.pid.to_numpy(), "base": base,
                                  "error": base & (night.psg_hum.to_numpy() != hb)}))
    return {"causal": np.vstack(parts[True]), "non_causal": np.vstack(parts[False]),
            "meta": pd.concat(meta, ignore_index=True)}


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make_model, n_splits: int = N_SPLITS) -> np.ndarray:
    """Predicted probability of error for every row, from a model that never saw that row's person."""
    scores = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model()
        model.fit(X[train], y[train])
        scores[test] = model.predict_proba(X[test])[:, 1]
    return scores


def kept_mask(scores: np.ndarray, coverage: float) -> np.ndarray:
    """The `coverage` share of epochs with the lowest scores (stable for ties)."""
    k = int(np.ceil(coverage * len(scores)))
    keep = np.zeros(len(scores), dtype=bool)
    keep[np.argsort(scores, kind="stable")[:k]] = True
    return keep


def accuracy_at_coverage(scores: np.ndarray, error: np.ndarray, coverage: float) -> float:
    keep = kept_mask(scores, coverage)
    return 1 - float(error[keep].mean()) if keep.any() else np.nan


def risk_coverage(scores: np.ndarray, error: np.ndarray, coverages=COVERAGE_GRID) -> list:
    """Accuracy on the kept epochs at each coverage, abstaining on the highest-scored epochs first."""
    order = np.argsort(scores, kind="stable")
    cum = np.cumsum(error[order].astype(float))
    out = []
    for c in coverages:
        k = int(np.ceil(c * len(scores)))
        out.append({"coverage": float(c), "accuracy": 1 - float(cum[k - 1]) / k})
    return out


def per_night_at_coverage(meta: pd.DataFrame, scores: np.ndarray, coverage: float) -> dict:
    """Per-night coverage and accuracy on kept epochs at a pooled coverage (nights with enough kept epochs)."""
    keep = kept_mask(scores, coverage)
    df = pd.DataFrame({"sub": meta["sub"].to_numpy(), "kept": keep, "err_kept": keep & meta.error.to_numpy()})
    g = df.groupby("sub", sort=False)
    n_base, n_kept, err = g.size(), g.kept.sum(), g.err_kept.sum()
    return {"coverage": summarise(n_kept / n_base),
            "accuracy_kept": summarise((1 - err / n_kept).where(n_kept >= MIN_KEPT_EPOCHS))}


def evaluate_model(scores: np.ndarray, meta: pd.DataFrame, heuristics: dict) -> dict:
    """AUROC, the risk-coverage curve, and the comparison with each hand-made flag at its own coverage."""
    error = meta.error.to_numpy()
    versus = {}
    for name, flag in heuristics.items():
        learned = accuracy_at_coverage(scores, error, flag["coverage"])
        versus[name] = {"coverage": flag["coverage"], "heuristic_accuracy": flag["accuracy_kept"],
                        "learned_accuracy": learned, "difference": learned - flag["accuracy_kept"],
                        "on_device": flag["on_device"]}
    return {"auroc": float(roc_auc_score(error, scores)), "average_precision": float(average_precision_score(error, scores)),
            "curve": risk_coverage(scores, error), "versus_hand_made_flags": versus,
            "per_night_at_change_1_coverage": per_night_at_coverage(meta, scores, heuristics[MATCHED_NIGHT_FLAG]["coverage"])}


def flag_auroc(table: pd.DataFrame, names: list[str]) -> dict:
    """AUROC of each hand-made flag read as a yes/no prediction of an error, for context."""
    df = ab.add_flags(table)
    base = both_stage(df).to_numpy()
    error = (df.psg_hum != df.hb_ai).to_numpy()[base]
    return {n: float(roc_auc_score(error, df[f"flag_{n}"].to_numpy()[base])) for n in names}


def run(table: pd.DataFrame, flag_sets: dict) -> dict:
    data = build_dataset(table)
    meta = data["meta"]
    base = meta.base.to_numpy()
    base_meta = meta[base].reset_index(drop=True)
    y, groups = base_meta.error.to_numpy().astype(int), base_meta.pid.to_numpy()
    heuristics = {n: flag_sets[n] for n in CONTEXT_FLAGS + ["combined_1", "change_2", "change_5"]}
    out = {"target": "headband and human both give a stage 0-4 and differ (human 8 excluded)",
           "n_base_epochs": int(base.sum()), "baseline_accuracy": 1 - float(y.mean()),
           "validation": f"{N_SPLITS}-fold, folds keep every night of one person together, scores are out of fold",
           "flag_auroc_hand_made": flag_auroc(table, [n for n in CONTEXT_FLAGS]), "feature_sets": {}}
    for fs, causal in (("causal", True), ("non_causal", False)):
        X = data[fs][base]
        out["feature_sets"][fs] = {"n_features": int(X.shape[1]), "models": {}}
        for name, make in MODELS.items():
            scores = oof_scores(X, y, groups, make)
            out["feature_sets"][fs]["models"][name] = evaluate_model(scores, base_meta, heuristics)
    return out


def fig_learned(result: dict, flag_sets: dict, out: Path):
    """Learned risk-coverage curves against the hand-made change flags, the PSG reference and chance."""
    base = 100 * result["baseline_accuracy"]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.axhline(base, color=MUTED, lw=1.0)
    ax.text(0.30, base + 0.25, f"chance: random abstention leaves it at {base:.1f} %", fontsize=8, color=MUTED,
            transform=ax.get_yaxis_transform())
    styles = {("non_causal", "gradient_boosting"): ("#1baf7a", "Learned, labels before and after (non-causal)"),
              ("causal", "gradient_boosting"): ("#e87ba4", "Learned, labels up to the epoch only (causal)")}
    for (fs, model), (color, label) in styles.items():
        curve = result["feature_sets"][fs]["models"][model]["curve"]
        ax.plot([100 * p["coverage"] for p in curve], [100 * p["accuracy"] for p in curve], color=color, lw=2.0,
                label=label)
    change = sorted((f["coverage"], f["accuracy_kept"], n) for n, f in flag_sets.items() if f["family"] == "change")
    ax.plot([100 * c[0] for c in change], [100 * c[1] for c in change], color="#2a78d6", lw=1.4, marker="o", ms=6,
            mec=SURFACE, mew=1.1, ls="--", label="Hand-made: within k epochs of a predicted change")
    psg = flag_sets["psg_ai_disagrees"]
    ax.plot(100 * psg["coverage"], 100 * psg["accuracy_kept"], marker="D", ms=9, mfc="none", mec=INK, mew=1.6,
            ls="none", label="PSG AI disagrees: NOT available on the device")
    ax.set_xlabel("Coverage: share of epochs the device keeps (%)")
    ax.set_ylabel("Agreement with the human on kept epochs (%)")
    ax.invert_xaxis()
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=INK2)
    ax.set_title("A small model on the labels alone against the hand-made flags")
    fig.text(0.01, 0.005, "Gradient-boosted model, out-of-fold, folds keep each person's nights together. "
             "Values: results/learned_abstention.json", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    flag_sets = json.loads(ABSTENTION.read_text())["flag_sets"]
    table = load_all(DATA)
    result = jsonable(run(table, flag_sets))
    RESULTS.write_text(json.dumps(result, indent=1))
    fig_learned(result, flag_sets, FIG_DIR / "fig16_learned_vs_heuristic.png")
    print(f"wrote {RESULTS} and {FIG_DIR}")


if __name__ == "__main__":
    main()
