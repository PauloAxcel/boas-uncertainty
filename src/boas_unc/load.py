"""Read the label files of the BOAS dataset and align them into one table per night.

Reads only the *_events.tsv and participants.tsv files in data/boas (never the EDFs).
Every epoch stays on the full timeline: special codes are flagged, never dropped
(docs/definitions.md, D2 and D8).

Columns of the table returned by load_night / load_all:
  sub               recording id (folder name, e.g. "sub-1"); one night, not one person
  epoch             0-based position of the epoch in time order
  onset             epoch start in seconds since the recording start (BIDS `onset`)
  psg_hum           human consensus label on the PSG (`stage_hum`), raw code
  psg_ai            AI label on the PSG, raw code
  hb_ai             AI label on the headband, raw code
  aligned           True when both the PSG file and the headband file have this onset
  hum_disconnected  psg_hum == 8 (PSG disconnected)
  psg_ai_missing    psg_ai == -2 (artefact / missing)
  hb_ai_missing     hb_ai == -2 (artefact / missing)
  hum_stage, psg_ai_stage, hb_ai_stage   W / N1 / N2 / N3 / REM, NaN for 8, -2 or no row
  pid               participant id from participants.tsv (load_all only)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Codes from data/boas/README, see docs/data_inventory.md section 5.
STAGE_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
HUM_DISCONNECTED = 8
AI_MISSING = -2

SOURCE_COLUMNS = ["psg_hum", "psg_ai", "hb_ai"]

# Age bands in years (inclusive), as used in checklist A8 and docs/definitions.md.
AGE_BANDS = [(18, 29), (30, 44), (45, 59), (60, 120)]


def subjects(data: Path) -> list[str]:
    """Recording folder names, sorted by their number."""
    return sorted((p.name for p in data.glob("sub-*") if p.is_dir()), key=lambda s: int(s[4:]))


def runs(seq: np.ndarray, code: int) -> list[tuple[int, int]]:
    """(start, length) of each run of `code` in seq."""
    hit = np.concatenate([[False], seq == code, [False]])
    edges = np.flatnonzero(np.diff(hit.astype(int)))
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def align_sources(psg: pd.DataFrame, hb: pd.DataFrame) -> pd.DataFrame:
    """Join the PSG and headband events on `onset`, not on row order.

    An onset present in only one file gives NaN in the other file's columns.
    Raises if either file repeats an onset.
    """
    night = psg[["onset", "stage_hum", "stage_ai"]].rename(
        columns={"stage_hum": "psg_hum", "stage_ai": "psg_ai"})
    night = night.merge(hb[["onset", "stage_ai"]].rename(columns={"stage_ai": "hb_ai"}),
                        on="onset", how="outer", validate="one_to_one")
    return night.sort_values("onset", kind="stable").reset_index(drop=True)


def stage_labels(codes: pd.Series) -> pd.Series:
    """Map stage codes to W/N1/N2/N3/REM. Codes 8 and -2, and missing values, become NaN."""
    return codes.map(STAGE_LABELS)


def flag_special(night: pd.DataFrame) -> pd.DataFrame:
    """Add the boolean flags for special codes and for epochs present in both files."""
    out = night.copy()
    out["aligned"] = night[SOURCE_COLUMNS].notna().all(axis=1)
    out["hum_disconnected"] = night.psg_hum == HUM_DISCONNECTED
    out["psg_ai_missing"] = night.psg_ai == AI_MISSING
    out["hb_ai_missing"] = night.hb_ai == AI_MISSING
    return out


def add_stage_labels(night: pd.DataFrame) -> pd.DataFrame:
    """Add hum_stage, psg_ai_stage and hb_ai_stage as W/N1/N2/N3/REM labels."""
    out = night.copy()
    for col, name in {"psg_hum": "hum_stage", "psg_ai": "psg_ai_stage", "hb_ai": "hb_ai_stage"}.items():
        out[name] = stage_labels(night[col])
    return out


def read_participants(data: Path) -> pd.DataFrame:
    """participants.tsv as strings, so values such as "n/a" are kept as they are."""
    return pd.read_csv(data / "participants.tsv", sep="\t", dtype=str, keep_default_na=False)


def attach_pid(table: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Add the `pid` column (same for all nights of one person) from participants.tsv."""
    pid = participants.set_index("participant_id").pid
    out = table.copy()
    out.insert(1, "pid", out["sub"].map(pid))
    return out


def read_events(data: Path, sub: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw PSG and headband events files of one night."""
    eeg = data / sub / "eeg"
    psg = pd.read_csv(eeg / f"{sub}_task-Sleep_acq-psg_events.tsv", sep="\t")
    hb = pd.read_csv(eeg / f"{sub}_task-Sleep_acq-headband_events.tsv", sep="\t")
    return psg, hb


def load_night(data: Path, sub: str) -> pd.DataFrame:
    """One row per epoch for one night, every epoch kept (see the module docstring)."""
    night = align_sources(*read_events(data, sub))
    night.insert(0, "epoch", np.arange(len(night)))
    night.insert(0, "sub", sub)
    return add_stage_labels(flag_special(night))


def load_all(data: Path) -> pd.DataFrame:
    """All nights stacked, with the participant id attached."""
    table = pd.concat([load_night(data, s) for s in subjects(data)], ignore_index=True)
    return attach_pid(table, read_participants(data))


def alignment_report(table: pd.DataFrame) -> pd.DataFrame:
    """Per night: epochs, epochs found in only one file, and whether the night is fully aligned."""
    g = table.groupby("sub", sort=False)
    report = pd.DataFrame({
        "n_epochs": g.size(),
        "n_psg_only": g.apply(lambda n: int(n.hb_ai.isna().sum()), include_groups=False),
        "n_headband_only": g.apply(lambda n: int(n.psg_hum.isna().sum()), include_groups=False),
    })
    report["aligned"] = (report.n_psg_only == 0) & (report.n_headband_only == 0)
    return report
