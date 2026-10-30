"""Wake, light and deep sleep: how well does the headband separate them? (D20)

Definitions: docs/definitions.md, "Three-class view for stimulation gating". Uses only the label files.
The mapping W -> wake, N1, N2 and REM -> light, N3 -> deep is an assumption about the stimulation classifier.
Writes results/three_class.json.

Run from the repo root:  uv run python -m boas_unc.three_class
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from boas_unc.agreement import is_stage, jsonable, summarise
from boas_unc.load import load_all

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
RESULTS = ROOT / "results" / "three_class.json"

CLASSES = ["wake", "light", "deep"]
MAPPING = {0: 0, 1: 1, 2: 1, 4: 1, 3: 2}  # stage code -> index in CLASSES (REM counted as light)
MIN_EPOCHS = 20


def to_three(codes: np.ndarray) -> np.ndarray:
    """Index of the three-class label for each stage code; -1 where the code is not a stage 0-4."""
    out = np.full(len(codes), -1)
    for code, k in MAPPING.items():
        out[codes == code] = k
    return out


def confusion(human: np.ndarray, headband: np.ndarray) -> np.ndarray:
    """3x3 counts: rows the human class, columns the headband class."""
    m = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            m[i, j] = int(((human == i) & (headband == j)).sum())
    return m


def precision_recall(m: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and recall of class k from a confusion matrix (NaN when undefined)."""
    tp, predicted, actual = m[k, k], m[:, k].sum(), m[k, :].sum()
    return (tp / predicted if predicted else np.nan), (tp / actual if actual else np.nan)


def f1(precision: float, recall: float) -> float:
    if np.isnan(precision) or np.isnan(recall):
        return np.nan
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def summarise_confusion(m: np.ndarray) -> dict:
    """Pooled metrics of one confusion matrix: accuracy, wake and deep precision and recall."""
    total = m.sum()
    rows = m.sum(axis=1, keepdims=True)
    out = {"n_epochs": int(total), "accuracy": float(np.trace(m) / total) if total else np.nan,
           "counts": {CLASSES[i]: {CLASSES[j]: int(m[i, j]) for j in range(3)} for i in range(3)},
           "row_normalised": {CLASSES[i]: {CLASSES[j]: (float(m[i, j] / rows[i, 0]) if rows[i, 0] else np.nan)
                                           for j in range(3)} for i in range(3)}}
    for k, name in ((0, "wake"), (2, "deep")):
        p, r = precision_recall(m, k)
        out[name] = {"precision": p, "recall": r, "f1": f1(p, r)}
    return out


def night_deep(human: np.ndarray, headband: np.ndarray) -> dict:
    """Deep recall and precision of one night, NaN where there are too few epochs."""
    m = confusion(human, headband)
    p, r = precision_recall(m, 2)
    return {"n_human_deep": int(m[2, :].sum()), "n_headband_deep": int(m[:, 2].sum()),
            "recall": r if m[2, :].sum() >= MIN_EPOCHS else np.nan,
            "precision": p if m[:, 2].sum() >= MIN_EPOCHS else np.nan}


def evaluate(table: pd.DataFrame, leave_out_rem: bool = False) -> dict:
    """Pooled and per-night results on the version (a) epochs, optionally without any REM epoch."""
    hum, hb = table.psg_hum.to_numpy(), table.hb_ai.to_numpy()
    base = is_stage(hum) & is_stage(hb)
    if leave_out_rem:
        base &= (hum != 4) & (hb != 4)
    h3, b3 = to_three(hum), to_three(hb)
    pooled = summarise_confusion(confusion(h3[base], b3[base]))
    say_deep = base & (b3 == 2) & (h3 != 2)
    n_say_deep = int((base & (b3 == 2)).sum())
    pooled["headband_deep_but_human_not_n3"] = {
        "share": int(say_deep.sum()) / n_say_deep if n_say_deep else np.nan,
        "human_says_n2": int((say_deep & (hum == 2)).sum()) / n_say_deep if n_say_deep else np.nan}
    # end-to-end: a human N3 epoch where the headband gives no stage is a missed deep epoch
    human_deep = is_stage(hum) & (h3 == 2)
    if leave_out_rem:
        human_deep &= (hb != 4)
    pooled["end_to_end_deep_recall"] = float((human_deep & (b3 == 2)).sum() / human_deep.sum()) if human_deep.any() else np.nan
    per = {}
    for sub, g in table.assign(_base=base).groupby("sub", sort=False):
        keep = g._base.to_numpy()
        per[sub] = night_deep(to_three(g.psg_hum.to_numpy())[keep], to_three(g.hb_ai.to_numpy())[keep])
    nights = pd.DataFrame(per).T
    eligible = nights[nights.n_human_deep >= MIN_EPOCHS]
    pooled["per_night"] = {"deep_recall": summarise(nights.recall.astype(float)),
                           "deep_precision": summarise(nights.precision.astype(float)),
                           "nights_with_enough_human_deep": int(len(eligible)),
                           "nights_where_headband_finds_none_of_them": int((eligible.recall.astype(float) == 0).sum())}
    return pooled


def run(table: pd.DataFrame) -> dict:
    return {"mapping": "wake = W; light = N1, N2, REM; deep = N3 (an assumption about the stimulation classifier)",
            "min_epochs_per_night": MIN_EPOCHS, "primary": evaluate(table), "without_rem": evaluate(table, True)}


def main():
    RESULTS.write_text(json.dumps(jsonable(run(load_all(DATA))), indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
