"""Agreement between label sequences (Milestone 3). Definitions: docs/definitions.md, D1 and D10.

Three pairs, each with a reference and a candidate:
  human_vs_psg_ai   psg_hum -> psg_ai
  human_vs_hb_ai    psg_hum -> hb_ai
  psg_ai_vs_hb_ai   psg_ai  -> hb_ai   (analysis tool only: the PSG AI is not on the device)

Epochs with human code 8 (PSG disconnected) are excluded from every version of every pair.
Two versions are computed for each pair:
  a_both_stage  epochs where reference and candidate are both stages 0-4 (Bitbrain-comparable)
  b_end_to_end  epochs where the reference is 0-4; a candidate -2 counts as an error
Coverage: share of reference 0-4 epochs where the candidate gives a stage 0-4.
Reference -2 epochs (PSG AI in pair 3) are never counted: there is no reference label there.

Run from the repo root:  uv run python -m boas_unc.agreement
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from boas_unc.load import AI_MISSING, STAGE_LABELS, load_all

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
RESULTS = ROOT / "results" / "agreement.json"

STAGES = list(STAGE_LABELS)  # [0, 1, 2, 3, 4]
PAIRS = {
    "human_vs_psg_ai": ("psg_hum", "psg_ai"),
    "human_vs_hb_ai": ("psg_hum", "hb_ai"),
    "psg_ai_vs_hb_ai": ("psg_ai", "hb_ai"),
}
VERSIONS = ("a_both_stage", "b_end_to_end")


def is_stage(codes: np.ndarray) -> np.ndarray:
    """True where the code is a sleep stage 0-4 (NaN, 8 and -2 are not)."""
    return np.isin(codes, STAGES)


def version_mask(ref: np.ndarray, cand: np.ndarray, version: str) -> np.ndarray:
    """Epochs that count in a version: both labels 0-4 (a), or reference 0-4 (b)."""
    if version == "a_both_stage":
        return is_stage(ref) & is_stage(cand)
    if version == "b_end_to_end":
        return is_stage(ref)
    raise ValueError(f"unknown version {version!r}")


def accuracy(ref: np.ndarray, cand: np.ndarray) -> float:
    """Share of epochs where the labels are equal. NaN with no epochs."""
    return float((ref == cand).mean()) if len(ref) else np.nan


def kappa(ref: np.ndarray, cand: np.ndarray) -> float:
    """Cohen's kappa; NaN when both sequences hold one and the same single class."""
    if len(np.unique(np.concatenate([ref, cand]))) < 2:
        return np.nan
    return float(cohen_kappa_score(ref, cand))


def coverage(ref: np.ndarray, cand: np.ndarray) -> float:
    """Share of reference 0-4 epochs where the candidate also gives a stage 0-4."""
    keep = is_stage(ref)
    return float(is_stage(cand[keep]).mean()) if keep.any() else np.nan


def per_stage_scores(ref: np.ndarray, cand: np.ndarray) -> dict:
    """Precision, recall, F1 and support per stage over labels 0-4 (D10).

    A candidate -2 is not one of the labels, so it lowers recall (a missed epoch) and leaves
    precision untouched. Recall is NaN when the reference has no such stage, precision is NaN
    when the candidate never predicts it. F1 is the count form 2TP / (2TP + FP + FN): NaN only
    when neither sequence has the stage, and 0 when the stage is present but entirely missed
    (or entirely false), so such nights are not dropped from the F1 distribution.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        ref, cand, labels=STAGES, zero_division=np.nan)
    return {STAGE_LABELS[s]: {"precision": float(p), "recall": float(r), "f1": float(f),
                              "support": int(n)}
            for s, p, r, f, n in zip(STAGES, precision, recall, f1, support)}


def confusion_counts(ref: np.ndarray, cand: np.ndarray) -> pd.DataFrame:
    """Rows: reference stage 0-4. Columns: candidate stage 0-4 and -2."""
    cols = STAGES + [AI_MISSING]
    ct = pd.crosstab(pd.Series(ref, name="ref"), pd.Series(cand, name="cand"))
    ct = ct.reindex(index=STAGES, columns=cols, fill_value=0)
    ct.index = [STAGE_LABELS[s] for s in STAGES]
    ct.columns = [STAGE_LABELS.get(c, str(c)) for c in cols]
    return ct


def row_normalise(counts: pd.DataFrame) -> pd.DataFrame:
    """Each row divided by its total; rows with no epochs are NaN."""
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)


def evaluate_epochs(ref: np.ndarray, cand: np.ndarray) -> dict:
    """All pooled metrics for one set of epochs."""
    counts = confusion_counts(ref, cand)
    return {"n_epochs": int(len(ref)), "n_agree": int((ref == cand).sum()),
            "accuracy": accuracy(ref, cand), "kappa": kappa(ref, cand),
            "per_stage": per_stage_scores(ref, cand),
            "confusion_counts": counts.to_dict(orient="index"),
            "confusion_row_normalised": row_normalise(counts).to_dict(orient="index")}


def summarise(values: pd.Series) -> dict:
    """Median and IQR over nights where the value is defined, with how many those are."""
    v = values.dropna()
    q25, med, q75 = np.percentile(v, [25, 50, 75]) if len(v) else (np.nan,) * 3
    return {"n_defined": int(len(v)), "n_undefined": int(len(values) - len(v)),
            "median": float(med), "q25": float(q25), "q75": float(q75)}


def night_scores(ref: np.ndarray, cand: np.ndarray) -> dict:
    """One night's accuracy, kappa and per-stage precision/recall/F1 (flat dict, NaN if undefined)."""
    out = {"accuracy": accuracy(ref, cand), "kappa": kappa(ref, cand)}
    for stage, scores in per_stage_scores(ref, cand).items():
        for metric in ("precision", "recall", "f1"):
            out[f"{stage}_{metric}"] = scores[metric]
    return out


def per_night(table: pd.DataFrame, ref_col: str, cand_col: str, version: str) -> pd.DataFrame:
    """Per-night scores for one pair and version. One row per night, NaN where undefined."""
    rows = {}
    for sub, night in table.groupby("sub", sort=False):
        ref, cand = night[ref_col].to_numpy(), night[cand_col].to_numpy()
        keep = version_mask(ref, cand, version)
        rows[sub] = night_scores(ref[keep], cand[keep])
    return pd.DataFrame.from_dict(rows, orient="index")


def evaluate_pair(table: pd.DataFrame, ref_col: str, cand_col: str) -> dict:
    """Pooled and per-night agreement for one pair, both versions, plus coverage."""
    table = table[~table.hum_disconnected]
    ref, cand = table[ref_col].to_numpy(), table[cand_col].to_numpy()
    out = {"reference": ref_col, "candidate": cand_col, "n_epochs_after_excluding_human_8": int(len(table))}
    for version in VERSIONS:
        keep = version_mask(ref, cand, version)
        scores = per_night(table, ref_col, cand_col, version)
        out[version] = {
            "pooled": evaluate_epochs(ref[keep], cand[keep]),
            "per_night": {col: summarise(scores[col]) for col in scores},
            "per_night_accuracy_by_night": scores["accuracy"].to_dict(),
        }
    by_night = table.groupby("sub", sort=False).apply(
        lambda n: coverage(n[ref_col].to_numpy(), n[cand_col].to_numpy()), include_groups=False)
    out["coverage"] = {"pooled": coverage(ref, cand), "per_night": summarise(by_night)}
    return out


def jsonable(x):
    """Recursively turn NaN into None and numpy numbers into Python ones (valid JSON)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 6)
    return x


def run(table: pd.DataFrame) -> dict:
    return {"n_nights": int(table["sub"].nunique()),
            "pairs": {name: evaluate_pair(table, ref, cand) for name, (ref, cand) in PAIRS.items()}}


def main():
    result = jsonable(run(load_all(DATA)))
    RESULTS.write_text(json.dumps(result, indent=1))
    for name, pair in result["pairs"].items():
        a, b = pair["a_both_stage"]["pooled"], pair["b_end_to_end"]["pooled"]
        print(f"{name}: (a) {a['n_agree']}/{a['n_epochs']} = {100 * a['accuracy']:.2f} %  "
              f"(b) {b['n_agree']}/{b['n_epochs']} = {100 * b['accuracy']:.2f} %")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
