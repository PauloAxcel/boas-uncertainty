"""How often is one channel clean while the headband gives no stage? (D18)

Definitions: docs/definitions.md, "One bad channel while the other is clean". Uses the per-epoch signal features
of signal_quality (complete files only, read-only) and the label files. It counts how often a clean channel was
available. It does not stage anything from one channel. Writes results/one_channel.json.

Run from the repo root:  uv run python -m boas_unc.one_channel
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from boas_unc import edf
from boas_unc import signal_quality as sq
from boas_unc.agreement import jsonable, summarise
from boas_unc.load import load_night

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "one_channel.json"
THRESHOLDS = {"primary": (10.0, 0.1), "sensitivity": (5.0, 0.2)}  # (high factor H, low factor L)
BOTH_GOOD, ONE_GOOD, BOTH_BAD = 0, 1, 2
STATUS_NAMES = {BOTH_GOOD: "both_good", ONE_GOOD: "one_good", BOTH_BAD: "both_bad"}


def reference_amplitude(std_uv: np.ndarray, staged: np.ndarray) -> float:
    """Median std over staged epochs with a non-zero std; NaN if there are none."""
    ok = staged & (std_uv > 0)
    return float(np.median(std_uv[ok])) if ok.any() else np.nan


def bad_channel(flat_share: np.ndarray, std_uv: np.ndarray, ref: float, high: float, low: float) -> np.ndarray:
    """True where the channel is flat, or at least `high` times or at most `low` times the reference amplitude."""
    return (flat_share >= sq.FLAT_EPOCH_SHARE) | (std_uv >= high * ref) | (std_uv <= low * ref)


def epoch_status(bad1: np.ndarray, bad2: np.ndarray) -> np.ndarray:
    """0 both channels good, 1 exactly one bad, 2 both bad."""
    return bad1.astype(int) + bad2.astype(int)


def night_counts(joined: pd.DataFrame, hum: np.ndarray, hb: np.ndarray, high: float, low: float) -> dict | None:
    """Everything D18 asks for one night, or None if the night has no reference amplitude."""
    c1, c2 = sq.CHANNELS
    refs = [reference_amplitude(joined[f"{c}_std_uv"].to_numpy(), joined.staged.to_numpy()) for c in (c1, c2)]
    if np.isnan(refs).any():
        return None
    bad = [bad_channel(joined[f"{c}_flat_share"].to_numpy(), joined[f"{c}_std_uv"].to_numpy(), r, high, low)
           for c, r in zip((c1, c2), refs)]
    status = epoch_status(*bad)
    miss, staged = joined.missing.to_numpy(), joined.staged.to_numpy()
    n = len(joined)
    both_stage = staged & np.isin(hum[:n], (0, 1, 2, 3, 4))
    agree = both_stage & (hum[:n] == hb[:n])
    out = {"n_epochs": int(n), "n_missing": int(miss.sum()), "references_uv": dict(zip((c1, c2), refs)),
           "missing_by_status": {STATUS_NAMES[k]: int((miss & (status == k)).sum()) for k in STATUS_NAMES},
           "staged_by_status": {STATUS_NAMES[k]: int((staged & (status == k)).sum()) for k in STATUS_NAMES},
           "one_good_missing_bad_channel": {c: int((miss & (status == ONE_GOOD) & b).sum()) for c, b in zip((c1, c2), bad)},
           "agreement_by_status": {STATUS_NAMES[k]: {"n": int((both_stage & (status == k)).sum()),
                                                      "n_agree": int((agree & (status == k)).sum())} for k in STATUS_NAMES}}
    one_good_missing = out["missing_by_status"]["one_good"]
    out["recoverable_share_of_missing"] = one_good_missing / out["n_missing"] if out["n_missing"] else None
    out["recoverable_minutes"] = one_good_missing * sq.EPOCH_S / 60
    return out


def aggregate(per_night: dict) -> dict:
    """Pooled counts and per-night distributions."""
    nights = {s: v for s, v in per_night.items() if v is not None}
    pooled = {k: {n: sum(v[k][n] for v in nights.values()) for n in STATUS_NAMES.values()}
              for k in ("missing_by_status", "staged_by_status")}
    miss = max(sum(pooled["missing_by_status"].values()), 1)
    one_good_all = pooled["missing_by_status"]["one_good"] + pooled["staged_by_status"]["one_good"]
    agree = {n: {"n": sum(v["agreement_by_status"][n]["n"] for v in nights.values()),
                 "n_agree": sum(v["agreement_by_status"][n]["n_agree"] for v in nights.values())}
             for n in STATUS_NAMES.values()}
    return {"n_nights": len(nights), "pooled": pooled,
            "share_of_missing_by_status": {n: c / miss for n, c in pooled["missing_by_status"].items()},
            "share_of_one_good_epochs_without_a_stage": (pooled["missing_by_status"]["one_good"] / one_good_all
                                                         if one_good_all else None),
            "one_good_missing_bad_channel": {c: sum(v["one_good_missing_bad_channel"][c] for v in nights.values())
                                             for c in sq.CHANNELS},
            "agreement": {n: {**a, "accuracy": a["n_agree"] / a["n"] if a["n"] else None} for n, a in agree.items()},
            "recoverable_share_per_night": summarise(pd.Series([v["recoverable_share_of_missing"] for v in nights.values()],
                                                               dtype=float)),
            "recoverable_minutes_per_night": summarise(pd.Series([v["recoverable_minutes"] for v in nights.values()
                                                                  if v["n_missing"] > 0], dtype=float)),
            "pooled_recoverable_minutes": pooled["missing_by_status"]["one_good"] * sq.EPOCH_S / 60,
            "pooled_missing_minutes": miss * sq.EPOCH_S / 60}


def run(data: Path, edf_dir: Path, subs: list[str] | None = None) -> dict:
    ok, skipped = sq.complete_nights(data, edf_dir, subs)
    per = {name: {} for name in THRESHOLDS}
    for sub in ok:
        path = sq.edf_path(edf_dir, sub)
        night = load_night(data, sub).sort_values("epoch").reset_index(drop=True)
        joined = sq.align(night, sq.night_quality(path, edf.read_header(path)))
        n = len(joined)
        for name, (high, low) in THRESHOLDS.items():
            per[name][sub] = night_counts(joined, night.psg_hum.to_numpy()[:n], night.hb_ai.to_numpy()[:n], high, low)
    return {"data_rule": "complete files only, read-only", "n_nights_skipped": len(skipped),
            "thresholds": {k: {"high_factor": h, "low_factor": l} for k, (h, l) in THRESHOLDS.items()},
            **{name: {**aggregate(p), "per_night": p} for name, p in per.items()}}


def main():
    result = jsonable(run(sq.DATA, sq.EDF_DIR))
    RESULTS.write_text(json.dumps(result, indent=1))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
