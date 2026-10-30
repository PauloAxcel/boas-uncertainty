"""Clinical sleep metrics from one label sequence. Definitions: docs/definitions.md (D2-D6, D8, D9, D11).

Every function takes one label sequence, the `onset` timestamps (seconds) where it needs time,
and a boolean mask of *valid* epochs. Excluded epochs (human code 8) are marked False in the mask;
the functions never look for code 8 themselves. Unscored epochs (headband -2) stay valid: they
count in time in bed and are neither sleep nor wake.

Latencies are timestamp differences on the full timeline, never row counts (D8).
Run the per-night comparison with `uv run python -m boas_unc.clinical`.
"""

import numpy as np

# One epoch is 30 s (data/boas *_events.json "always 30", docs/data_inventory.md section 6).
EPOCH_MIN = 0.5

WAKE = 0
REM = 4
SLEEP_CODES = (1, 2, 3, 4)
UNSCORED = -2
ALLOWED_CODES = (0, 1, 2, 3, 4, UNSCORED)
STAGE_NAMES = {1: "n1", 2: "n2", 3: "n3", 4: "rem"}
MIN_REM_RUN = 3  # epochs; sustained REM latency sensitivity check (D13)


def check_codes(seq: np.ndarray, valid: np.ndarray) -> None:
    """Raise if a valid epoch holds a code other than 0-4 or -2 (including NaN)."""
    bad = valid & ~np.isin(seq, ALLOWED_CODES)
    if bad.any():
        raise ValueError(f"unexpected codes on valid epochs: {sorted(set(seq[bad].tolist()))}")


def sleep_epochs(seq: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """True on valid epochs that are N1, N2, N3 or REM."""
    return valid & np.isin(seq, SLEEP_CODES)


def time_in_bed(valid: np.ndarray) -> float:
    """TIB in minutes: every valid epoch, so human 8 is out and headband -2 stays in."""
    return float(valid.sum()) * EPOCH_MIN


def total_sleep_time(seq: np.ndarray, valid: np.ndarray) -> float:
    """TST in minutes."""
    return float(sleep_epochs(seq, valid).sum()) * EPOCH_MIN


def sleep_efficiency(tst: float, tib: float) -> float:
    """100 x TST / TIB. NaN if the time in bed is zero."""
    return 100 * tst / tib if tib > 0 else np.nan


def first_sleep_index(seq: np.ndarray, valid: np.ndarray) -> int | None:
    """Index of the sleep onset epoch (first valid sleep epoch), None if there is no sleep."""
    idx = np.flatnonzero(sleep_epochs(seq, valid))
    return int(idx[0]) if len(idx) else None


def last_sleep_index(seq: np.ndarray, valid: np.ndarray) -> int | None:
    """Index of the last valid sleep epoch, None if there is no sleep."""
    idx = np.flatnonzero(sleep_epochs(seq, valid))
    return int(idx[-1]) if len(idx) else None


def left_censored(seq: np.ndarray, valid: np.ndarray) -> bool:
    """True when the recording's first epoch is valid and already sleep: the real onset is unknown (D4)."""
    return len(seq) > 0 and bool(valid[0] and np.isin(seq[0], SLEEP_CODES))


def sleep_onset_latency(seq: np.ndarray, onset: np.ndarray, valid: np.ndarray) -> float:
    """(onset of sleep onset epoch - onset of the first epoch) / 60. NaN if censored or no sleep."""
    i = first_sleep_index(seq, valid)
    if i is None or left_censored(seq, valid):
        return np.nan
    return float(onset[i] - onset[0]) / 60


def rem_latency(seq: np.ndarray, onset: np.ndarray, valid: np.ndarray) -> float:
    """(onset of first REM epoch - onset of sleep onset epoch) / 60. NaN if censored or no REM."""
    i = first_sleep_index(seq, valid)
    rem = np.flatnonzero(valid & (seq == REM))
    if i is None or not len(rem) or left_censored(seq, valid):
        return np.nan
    return float(onset[rem[0]] - onset[i]) / 60


def first_run_start(mask: np.ndarray, min_len: int) -> int | None:
    """Index where the first run of at least min_len consecutive True values starts, None if none."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= min_len:
            return int(start)
    return None


def sustained_rem_latency(seq: np.ndarray, onset: np.ndarray, valid: np.ndarray, min_run: int = MIN_REM_RUN) -> float:
    """REM latency to the first run of at least min_run consecutive valid REM epochs (D13 sensitivity).

    Measured from the sleep onset epoch. NaN if the night is censored, has no sleep, or has no such run
    (even if it has shorter REM stretches). An excluded epoch breaks a run.
    """
    i = first_sleep_index(seq, valid)
    start = first_run_start(valid & (seq == REM), min_run)
    if i is None or start is None or left_censored(seq, valid):
        return np.nan
    return float(onset[start] - onset[i]) / 60


def pre_sleep_wake(seq: np.ndarray, valid: np.ndarray) -> float:
    """Minutes of Wake before sleep onset. NaN if there is no sleep; 0 on a censored night."""
    i = first_sleep_index(seq, valid)
    return np.nan if i is None else float((valid & (seq == WAKE))[:i].sum()) * EPOCH_MIN


def waso(seq: np.ndarray, valid: np.ndarray) -> float:
    """Minutes of Wake between sleep onset and the last sleep epoch. NaN if there is no sleep."""
    i, j = first_sleep_index(seq, valid), last_sleep_index(seq, valid)
    return np.nan if i is None else float((valid & (seq == WAKE))[i:j + 1].sum()) * EPOCH_MIN


def terminal_wake(seq: np.ndarray, valid: np.ndarray) -> float:
    """Minutes of Wake after the last sleep epoch to the end of the recording."""
    j = last_sleep_index(seq, valid)
    return np.nan if j is None else float((valid & (seq == WAKE))[j + 1:].sum()) * EPOCH_MIN


def unscored_time(seq: np.ndarray, valid: np.ndarray) -> float:
    """Minutes of headband -2 on valid epochs."""
    return float((valid & (seq == UNSCORED)).sum()) * EPOCH_MIN


def stage_percentages(seq: np.ndarray, valid: np.ndarray) -> dict:
    """pct_n1 ... pct_rem: 100 x epochs of the stage / TST. They sum to 100. NaN if TST is 0."""
    n_sleep = sleep_epochs(seq, valid).sum()
    return {f"pct_{name}": 100 * float((valid & (seq == code)).sum()) / n_sleep if n_sleep else np.nan
            for code, name in STAGE_NAMES.items()}


def sleep_metrics(seq: np.ndarray, onset: np.ndarray, valid: np.ndarray) -> dict:
    """All standalone metrics (analysis (i)) of one sequence on its full timeline."""
    check_codes(seq, valid)
    tib, tst = time_in_bed(valid), total_sleep_time(seq, valid)
    tw = terminal_wake(seq, valid)
    w = waso(seq, valid)
    return {"tib": tib, "tst": tst, "sleep_efficiency": sleep_efficiency(tst, tib),
            "sol": sleep_onset_latency(seq, onset, valid), "pre_sleep_wake": pre_sleep_wake(seq, valid),
            "rem_latency": rem_latency(seq, onset, valid),
            "rem_latency_sustained": sustained_rem_latency(seq, onset, valid), "waso": w, "terminal_wake": tw,
            "waso_incl_terminal": w + tw, "unscored": unscored_time(seq, valid),
            **stage_percentages(seq, valid), "left_censored": left_censored(seq, valid)}


def matched_mask(hum: np.ndarray, other: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Valid epochs where both sequences give a stage 0-4 (analysis (ii))."""
    return valid & np.isin(hum, (0, 1, 2, 3, 4)) & np.isin(other, (0, 1, 2, 3, 4))


def matched_metrics(seq: np.ndarray, matched: np.ndarray) -> dict:
    """Stage-based metrics of one sequence on the matched epochs only (analysis (ii)).

    No latencies, WASO or terminal wake: masking changes which epoch counts as first (D8).
    """
    tib_matched, tst = time_in_bed(matched), total_sleep_time(seq, matched)
    return {"tib_matched": tib_matched, "tst": tst,
            "sleep_efficiency": sleep_efficiency(tst, tib_matched), **stage_percentages(seq, matched)}
