"""What does the raw headband signal look like when the headband gives no stage? (D17)

Definitions: docs/definitions.md, "Signal quality of the headband". Reads only complete headband EDF
files (size and SHA256 match the git-annex key in data/boas), read-only, from the folder in the
BOAS_EDF_DIR environment variable (default: headband_edfs/ in the repo root). Writes results/signal_quality.json.

Run from the repo root:  uv run python -m boas_unc.signal_quality
"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from sklearn.metrics import roc_auc_score

from boas_unc import edf
from boas_unc.agreement import jsonable, summarise
from boas_unc.hotspots import distance_to_nearest
from boas_unc.load import AI_MISSING, load_night, read_participants, runs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
EDF_DIR = Path(os.environ.get("BOAS_EDF_DIR", ROOT / "headband_edfs"))
RESULTS = ROOT / "results" / "signal_quality.json"

EPOCH_S = 30.0
CHANNELS = ["HB_1", "HB_2"]
ACCEL = ["HB_IMU_1", "HB_IMU_2", "HB_IMU_3"]
FLAT_EPOCH_SHARE = 0.5   # an epoch is flat when both channels have at least this flat_share
SAT_FRACTION = 0.99
TOTAL_BAND, HF_BAND, MAINS_BAND = (0.5, 45.0), (30.0, 45.0), (49.0, 51.0)
WELCH_S = 2.0
LONG_RUN = 10   # epochs
BEFORE = 5      # epochs before a run
NEAR, FAR = 5, 30
FEATURES = ["flat_share", "std_uv", "p2p_uv", "sat_share", "hf_ratio", "mains_ratio"]


# ---------------------------------------------------------------- file integrity

def annex_key(data: Path, sub: str) -> tuple[int, str]:
    """(size, sha256) of a night's headband EDF from its git-annex symlink in the dataset clone."""
    link = os.readlink(data / sub / "eeg" / f"{sub}_task-Sleep_acq-headband_eeg.edf")
    m = re.search(r"SHA256E-s(\d+)--([0-9a-f]{64})", link)
    if not m:
        raise ValueError(f"no SHA256E annex key in {link}")
    return int(m.group(1)), m.group(2)


def edf_path(edf_dir: Path, sub: str) -> Path:
    return edf_dir / sub / "eeg" / f"{sub}_task-Sleep_acq-headband_eeg.edf"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def is_complete(path: Path, key: tuple[int, str]) -> bool:
    """True when the file exists, has the expected size, and the expected SHA256 (size is checked first)."""
    return path.is_file() and path.stat().st_size == key[0] and sha256_file(path) == key[1]


def complete_nights(data: Path, edf_dir: Path, subs: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Nights whose file is complete, and nights whose file is missing or still downloading."""
    from boas_unc.load import subjects
    ok, skipped = [], []
    for sub in subs or subjects(data):
        (ok if is_complete(edf_path(edf_dir, sub), annex_key(data, sub)) else skipped).append(sub)
    return ok, skipped


# ---------------------------------------------------------------- features

def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    keep = (freqs >= band[0]) & (freqs <= band[1])
    return psd[:, keep].sum(axis=1) * (freqs[1] - freqs[0])


def constant_epochs(ep: np.ndarray) -> np.ndarray:
    """Epochs in which every sample is identical. Exact, so a signal stuck at a non-zero level counts too."""
    return np.ptp(ep, axis=1) == 0


def spectral_ratios(ep: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """hf_ratio and mains_ratio of each epoch (NaN where the epoch is constant: it has no power, only rounding noise)."""
    freqs, psd = sps.welch(ep, fs=fs, nperseg=int(WELCH_S * fs), noverlap=0, axis=1)
    total = np.where(constant_epochs(ep), 0.0, band_power(freqs, psd, TOTAL_BAND))
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = np.where(total > 0, band_power(freqs, psd, HF_BAND) / total, np.nan)
        mains = np.where(total > 0, band_power(freqs, psd, MAINS_BAND) / total, np.nan)
    return hf, mains


def epoch_features(ep: np.ndarray, fs: float, limit_uv: float) -> pd.DataFrame:
    """The D17 features of every epoch of one channel (ep: epochs x samples, microvolts)."""
    hf, mains = spectral_ratios(ep, fs)
    return pd.DataFrame({"flat_share": (np.diff(ep, axis=1) == 0).mean(axis=1),
                         "std_uv": np.where(constant_epochs(ep), 0.0, ep.std(axis=1)),
                         "p2p_uv": np.ptp(ep, axis=1),
                         "sat_share": (np.abs(ep) >= SAT_FRACTION * limit_uv).mean(axis=1),
                         "hf_ratio": hf, "mains_ratio": mains})


def night_quality(path: Path, header: dict | None = None) -> pd.DataFrame:
    """Per-epoch features of both channels (columns HB_1_flat_share, ...) and `flat_epoch`."""
    header = header or edf.read_header(path)
    cols = {}
    for ch in CHANNELS:
        sig = header["signals"][edf.signal_index(header, ch)]
        ep = edf.epochs(edf.read_signal(path, ch, header), sig["sampling_hz"], EPOCH_S)
        feats = epoch_features(ep, sig["sampling_hz"], abs(sig["physical_max"]))
        cols.update({f"{ch}_{f}": feats[f].to_numpy() for f in FEATURES})
    out = pd.DataFrame(cols)
    out["flat_epoch"] = (out[[f"{ch}_flat_share" for ch in CHANNELS]] >= FLAT_EPOCH_SHARE).all(axis=1)
    return out


def movement_per_epoch(path: Path, header: dict | None = None) -> np.ndarray | None:
    """Standard deviation of the accelerometer vector norm in each epoch, or None without movement signals."""
    header = header or edf.read_header(path)
    labels = [s["label"] for s in header["signals"]]
    if not all(a in labels for a in ACCEL):
        return None
    axes = [edf.read_signal(path, a, header) for a in ACCEL]
    fs = header["signals"][edf.signal_index(header, ACCEL[0])]["sampling_hz"]
    norm = np.sqrt(sum(a.astype(float) ** 2 for a in axes))
    return edf.epochs(norm, fs, EPOCH_S).std(axis=1)


# ---------------------------------------------------------------- the questions

def align(night: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Label epochs and signal epochs side by side. Raises if their numbers differ by more than the partial epoch."""
    n = min(len(night), len(quality))
    if abs(len(night) - len(quality)) > 1:
        raise ValueError(f"label file has {len(night)} epochs, signal has {len(quality)}")
    out = quality.iloc[:n].reset_index(drop=True)
    out.insert(0, "missing", (night.hb_ai.to_numpy()[:n] == AI_MISSING))
    out.insert(1, "staged", np.isin(night.hb_ai.to_numpy()[:n], (0, 1, 2, 3, 4)))
    return out


def night_q1(joined: pd.DataFrame) -> dict:
    """Overlap of `-2` epochs and flat epochs in one night, and feature medians by group."""
    miss, flat = joined.missing.to_numpy(), joined.flat_epoch.to_numpy()
    cols = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURES]
    return {"n_epochs": int(len(joined)), "n_missing": int(miss.sum()), "n_flat": int(flat.sum()),
            "share_of_missing_that_are_flat": float((flat & miss).sum() / miss.sum()) if miss.any() else None,
            "share_of_flat_that_are_missing": float((flat & miss).sum() / flat.sum()) if flat.any() else None,
            "median_when_missing": joined.loc[joined.missing, cols].median().to_dict(),
            "median_when_staged": joined.loc[joined.staged, cols].median().to_dict()}


def pooled_auroc(frames: list[pd.DataFrame]) -> dict:
    """AUROC of each feature for telling `-2` epochs (positive) from staged epochs, pooled over nights."""
    df = pd.concat(frames, ignore_index=True)
    df = df[df.missing | df.staged]
    out = {}
    for col in [f"{ch}_{f}" for ch in CHANNELS for f in FEATURES]:
        ok = df[col].notna()
        y = df.missing[ok].to_numpy()
        out[col] = float(roc_auc_score(y, df[col][ok])) if 0 < y.sum() < len(y) else None
    return {"n_epochs": int(len(df)), "n_missing": int(df.missing.sum()), "auroc_by_feature": out}


def near_far_staged(joined: pd.DataFrame) -> dict:
    """Median features of staged epochs within NEAR epochs of a `-2` and those at least FAR epochs from any."""
    d = distance_to_nearest(joined.missing.to_numpy())
    cols = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURES]
    near = joined[joined.staged.to_numpy() & (d <= NEAR) & (d >= 1)]
    far = joined[joined.staged.to_numpy() & (d >= FAR)]
    return {"n_near": int(len(near)), "n_far": int(len(far)),
            "median_near": near[cols].median().to_dict() if len(near) else None,
            "median_far": far[cols].median().to_dict() if len(far) else None}


def movement_ratios(missing: np.ndarray, staged: np.ndarray, movement: np.ndarray) -> list[float]:
    """For each long `-2` run: movement just before it over movement of staged epochs far from any `-2`."""
    n = min(len(missing), len(movement))
    missing, staged, movement = missing[:n], staged[:n], movement[:n]
    far = staged & (distance_to_nearest(missing) >= FAR)
    if not far.any():
        return []
    baseline = float(movement[far].mean())
    out = []
    for start, length in runs(missing.astype(int), 1):
        if length >= LONG_RUN and start >= BEFORE and staged[start - BEFORE:start].all() and baseline > 0:
            out.append(float(movement[start - BEFORE:start].mean()) / baseline)
    return out


def movement_ranks(missing: np.ndarray, staged: np.ndarray, movement: np.ndarray) -> list[float]:
    """For each long `-2` run: the share of comparable windows quieter than the one before the run.

    The comparable windows are the BEFORE-epoch windows made only of staged epochs at least FAR epochs from any
    `-2` epoch. A rank of 0.5 means the run was preceded by typical movement. The mean-based ratio of
    movement_ratios is pulled down by spikes in the baseline, so this rank is the fairer comparison.
    """
    n = min(len(missing), len(movement))
    missing, staged, movement = missing[:n], staged[:n], movement[:n].astype(float)
    far = staged & (distance_to_nearest(missing) >= FAR)
    if n < BEFORE:
        return []
    kernel = np.ones(BEFORE)
    window_mean = np.convolve(movement, kernel / BEFORE, mode="valid")  # window i covers epochs i .. i + BEFORE - 1
    window_far = np.convolve(far.astype(int), kernel, mode="valid") == BEFORE
    baseline = window_mean[window_far]
    if not len(baseline):
        return []
    out = []
    for start, length in runs(missing.astype(int), 1):
        if length >= LONG_RUN and start >= BEFORE and staged[start - BEFORE:start].all():
            out.append(float((baseline <= movement[start - BEFORE:start].mean()).mean()))
    return out


def order_check(starts: dict[str, pd.Timestamp], pid_of: dict[str, str]) -> list[dict]:
    """For each person with two or more complete nights: the order by EDF start and by sub number."""
    people = {}
    for sub, start in starts.items():
        people.setdefault(pid_of[sub], []).append((sub, start))
    out = []
    for pid, nights in people.items():
        if len(nights) >= 2:
            by_date = [s for s, _ in sorted(nights, key=lambda x: x[1])]
            by_number = sorted((s for s, _ in nights), key=lambda s: int(s.split("-")[1]))
            out.append({"pid": pid, "by_start": by_date, "by_number": by_number, "same_order": by_date == by_number,
                        "starts": {s: str(t) for s, t in nights}})
    return out


def run(data: Path, edf_dir: Path, subs: list[str] | None = None) -> dict:
    ok, skipped = complete_nights(data, edf_dir, subs)
    participants = read_participants(data).set_index("participant_id")
    joined, per_night, near_far, ratios, ranks, starts, moved = {}, {}, {}, {}, {}, {}, []
    for sub in ok:
        path = edf_path(edf_dir, sub)
        header = edf.read_header(path)
        starts[sub] = pd.Timestamp(header["start"])
        night = load_night(data, sub).sort_values("epoch")
        joined[sub] = align(night, night_quality(path, header))
        per_night[sub] = night_q1(joined[sub])
        near_far[sub] = near_far_staged(joined[sub])
        mv = movement_per_epoch(path, header)
        if mv is not None:
            moved.append(sub)
            ratios[sub] = movement_ratios(joined[sub].missing.to_numpy(), joined[sub].staged.to_numpy(), mv)
            ranks[sub] = movement_ranks(joined[sub].missing.to_numpy(), joined[sub].staged.to_numpy(), mv)
    all_ratios = [r for v in ratios.values() for r in v]
    all_ranks = [r for v in ranks.values() for r in v]
    return {"data_rule": "complete files only (size and SHA256 match the git-annex key), read-only",
            "n_nights_used": len(ok), "nights_used": ok, "n_nights_skipped_incomplete_or_missing": len(skipped),
            "q1_per_night": per_night,
            "q1_pooled": pooled_auroc(list(joined.values())) if joined else None,
            "q2_movement_before_long_runs": {"nights_with_movement_signals": moved, "n_runs": len(all_ratios),
                                             "ratio": summarise(pd.Series(all_ratios)),
                                             "n_ratio_above_1": int(sum(r > 1 for r in all_ratios)),
                                             "rank": summarise(pd.Series(all_ranks, dtype=float)),
                                             "n_rank_above_half": int(sum(r > 0.5 for r in all_ranks))},
            "q3_order_of_nights": order_check(starts, participants.pid.to_dict()),
            "q4_staged_near_vs_far_from_missing": near_far}


def main():
    result = jsonable(run(DATA, EDF_DIR))
    RESULTS.write_text(json.dumps(result, indent=1))
    print(f"nights used {result['n_nights_used']}, skipped {result['n_nights_skipped_incomplete_or_missing']}; wrote {RESULTS}")


if __name__ == "__main__":
    main()
