"""Where do the human and the headband AI disagree? (Milestone 4). Definitions: docs/definitions.md, D12.

Reference: human (psg_hum). Candidate: headband AI (hb_ai). Human 8 epochs are excluded.
Disagreement rate = 1 - accuracy over epochs where both labels are stages 0-4 (version (a), D1).
The headband -2 effect is reported next to it as the missing rate, never inside the rate.

Run from the repo root:  uv run python -m boas_unc.hotspots
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boas_unc.agreement import confusion_counts, is_stage, jsonable, row_normalise, summarise
from boas_unc.inventory_figures import GRID, INK, INK2, MUTED, SURFACE, style
from boas_unc.load import AGE_BANDS, AI_MISSING, STAGE_LABELS, load_all, read_participants

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
RESULTS = ROOT / "results" / "hotspots.json"
FIG_DIR = ROOT / "figures" / "hotspots"

MAX_K = 5            # distances 0..5, then one group "6+"
FLAG_MAX_D = 5       # artefact distances 1..5
HOUR_CAP = 9         # hours 0..8, then one group "9+"
MIN_NIGHT_EPOCHS = 10  # a night enters a group's per-night summary with at least this many epochs
MIN_PLOT_EPOCHS = 50   # figure: points from groups with fewer epochs are not drawn

# Reference palette, categorical slots 1-5 (validated in light mode). Colour follows the stage.
STAGE_COLORS = {"W": "#2a78d6", "N1": "#eb6834", "N2": "#1baf7a", "N3": "#eda100", "REM": "#e87ba4"}
STAGE_MARKERS = {"W": "o", "N1": "s", "N2": "^", "N3": "D", "REM": "v"}


def distance_to_next(mask: np.ndarray) -> np.ndarray:
    """Epochs from each position to the next True at or after it; inf if there is none."""
    pos = np.flatnonzero(mask)
    idx = np.arange(len(mask))
    if not len(pos):
        return np.full(len(mask), np.inf)
    j = np.searchsorted(pos, idx, side="left")
    return np.where(j < len(pos), pos[np.minimum(j, len(pos) - 1)] - idx, np.inf)


def distance_from_prev(mask: np.ndarray) -> np.ndarray:
    """Epochs from the last True at or before each position to it; inf if there is none."""
    pos = np.flatnonzero(mask)
    idx = np.arange(len(mask))
    if not len(pos):
        return np.full(len(mask), np.inf)
    j = np.searchsorted(pos, idx, side="right") - 1
    return np.where(j >= 0, idx - pos[np.maximum(j, 0)], np.inf)


def distance_to_nearest(mask: np.ndarray) -> np.ndarray:
    """Epochs to the closest True in either direction (0 on a True); inf if there is none."""
    return np.minimum(distance_to_next(mask), distance_from_prev(mask))


def transition_mask(hum: np.ndarray) -> np.ndarray:
    """True on epochs with a stage 0-4 whose neighbour is a different stage 0-4 (D12).

    A boundary into or out of code 8 is not a transition.
    """
    stage = is_stage(hum)
    differs_prev = np.zeros(len(hum), dtype=bool)
    differs_next = np.zeros(len(hum), dtype=bool)
    differs_prev[1:] = stage[1:] & stage[:-1] & (hum[1:] != hum[:-1])
    differs_next[:-1] = stage[:-1] & stage[1:] & (hum[:-1] != hum[1:])
    return differs_prev | differs_next


def cap_distance(d: np.ndarray, max_k: int = MAX_K) -> np.ndarray:
    """Integer distance, with everything beyond max_k (and inf) put in the group max_k + 1."""
    return np.minimum(np.where(np.isinf(d), max_k + 1, d), max_k + 1).astype(int)


def hour_of_night(onset: np.ndarray, cap: int = HOUR_CAP) -> np.ndarray:
    """Whole hours since the recording start, with every hour from `cap` on in one group."""
    return np.minimum(np.floor(np.asarray(onset) / 3600), cap).astype(int)


def annotate_night(night: pd.DataFrame) -> pd.DataFrame:
    """Add distance to the nearest human transition, distances to headband -2 flags, and hour."""
    out = night.copy()
    flag = night.hb_ai.to_numpy() == AI_MISSING
    out["dist_transition"] = cap_distance(distance_to_nearest(transition_mask(night.psg_hum.to_numpy())))
    out["to_next_flag"] = distance_to_next(flag)
    out["since_prev_flag"] = distance_from_prev(flag)
    out["hour"] = hour_of_night(night.onset.to_numpy())
    return out


def annotate(table: pd.DataFrame) -> pd.DataFrame:
    """annotate_night for every night (rows must be in time order within a night)."""
    return pd.concat([annotate_night(n) for _, n in table.groupby("sub", sort=False)], ignore_index=True)


def both_stage(df: pd.DataFrame) -> pd.Series:
    """Version (a) epochs: human and headband both give a stage 0-4 (human 8 is never one)."""
    return pd.Series(is_stage(df.psg_hum.to_numpy()) & is_stage(df.hb_ai.to_numpy()), index=df.index)


def rate_summary(df: pd.DataFrame) -> dict:
    """Pooled disagreement and missing rates of a set of epochs (D12)."""
    human = is_stage(df.psg_hum.to_numpy())
    both = both_stage(df).to_numpy()
    n_both, n_human = int(both.sum()), int(human.sum())
    n_dis = int((df.psg_hum.to_numpy()[both] != df.hb_ai.to_numpy()[both]).sum())
    n_missing = int((human & (df.hb_ai.to_numpy() == AI_MISSING)).sum())
    return {"n_epochs": n_both, "n_disagree": n_dis,
            "disagreement_rate": n_dis / n_both if n_both else np.nan,
            "n_human_epochs": n_human, "headband_missing_rate": n_missing / n_human if n_human else np.nan,
            "n_nights": int(df.loc[both, "sub"].nunique())}


def night_rates(df: pd.DataFrame, min_epochs: int = MIN_NIGHT_EPOCHS) -> pd.Series:
    """Disagreement rate per night; NaN for nights with fewer than min_epochs version (a) epochs.

    Indexed over the nights that have at least one version (a) epoch in `df`, the same nights
    that rate_summary counts in n_nights, so n_defined + n_undefined == n_nights in every group.
    """
    work = df[both_stage(df)]
    g = work.groupby("sub", sort=False)
    n, dis = g.size(), (work.psg_hum != work.hb_ai).groupby(work["sub"], sort=False).sum()
    return (dis / n).where(n >= min_epochs)


def group_summary(df: pd.DataFrame) -> dict:
    """Pooled rates plus the per-night median and IQR for one group of epochs."""
    return {**rate_summary(df), "per_night_rate": summarise(night_rates(df))}


def by_key(df: pd.DataFrame, key: str, names: dict | None = None) -> dict:
    """group_summary for each value of a column. `names` relabels the values."""
    return {(names or {}).get(k, str(k)): group_summary(g) for k, g in df.groupby(key, sort=True)}


def distance_label(k: int) -> str:
    return f"{k}" if k <= MAX_K else f"{MAX_K + 1}+"


def hour_label(h: int) -> str:
    return f"{h}" if h < HOUR_CAP else f"{HOUR_CAP}+"


DISTANCE_ORDER = [distance_label(k) for k in range(MAX_K + 2)]


def by_true_stage(df: pd.DataFrame) -> dict:
    """Disagreement by human stage, with where the disagreements go (row-normalised confusion)."""
    df = df.assign(stage=df.psg_hum.map(STAGE_LABELS))
    out = {s: group_summary(g) for s, g in df[df.stage.notna()].groupby("stage", sort=False)}
    out = {s: out[s] for s in STAGE_LABELS.values() if s in out}
    both = both_stage(df).to_numpy()
    counts = confusion_counts(df.psg_hum.to_numpy()[both], df.hb_ai.to_numpy()[both])
    return {"by_stage": out, "confusion_counts_version_a": counts.to_dict(orient="index"),
            "confusion_row_normalised_version_a": row_normalise(counts).to_dict(orient="index")}


def by_distance_to_transition(df: pd.DataFrame) -> dict:
    labelled = df.assign(label=df.dist_transition.map(distance_label))
    return {k: group_summary(labelled[labelled.label == k]) for k in DISTANCE_ORDER
            if (labelled.label == k).any()}


def stage_by_distance(df: pd.DataFrame) -> dict:
    """Pooled rates for every human stage and distance bin: the data behind the figure."""
    out = {}
    for code, stage in STAGE_LABELS.items():
        sub = df[df.psg_hum == code]
        out[stage] = {distance_label(k): rate_summary(sub[sub.dist_transition == k])
                      for k in range(MAX_K + 2)}
    return out


def by_artefact_distance(df: pd.DataFrame) -> dict:
    """Rates just before and just after a headband -2 flag, and away from any flag (D12)."""
    near = (df.to_next_flag <= FLAG_MAX_D) | (df.since_prev_flag <= FLAG_MAX_D)
    out = {"no_flag_within_5": group_summary(df[~near]), "before_flag": {}, "after_flag": {}}
    for d in range(1, FLAG_MAX_D + 1):
        out["before_flag"][str(d)] = group_summary(df[df.to_next_flag == d])
        out["after_flag"][str(d)] = group_summary(df[df.since_prev_flag == d])
    return out


def per_person(rates: pd.Series, pid_of: pd.Series) -> pd.Series:
    """Mean of each person's night rates (D7)."""
    return rates.groupby(pid_of.reindex(rates.index)).mean()


def person_groups(rates: pd.Series, pid_of: pd.Series, group_of: pd.Series, order: list[str]) -> dict:
    """Per group (age band or sex): nights, people, and the person-level rate distribution (D7).

    Nights whose group or rate is missing are left out. A person's rate in a group is the mean
    over that person's nights in the group. Groups come out in the order given, skipping empty ones.
    """
    frame = pd.DataFrame({"rate": rates, "pid": pid_of.reindex(rates.index),
                          "group": group_of.reindex(rates.index)}).dropna(subset=["group", "rate"])
    return {group: {"n_nights": int(len(g)), "n_pids": int(g.pid.nunique()),
                    "per_person_rate": summarise(g.groupby("pid").rate.mean())}
            for group in order if len(g := frame[frame.group == group])}


def band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < 120 else f"{lo}+"


AGE_BAND_ORDER = [band_label(lo, hi) for lo, hi in AGE_BANDS]
SEX_ORDER = ["F", "M"]


def age_band(age: str) -> str | None:
    """Age band label for an age string, None for "n/a" or anything that isn't a number."""
    try:
        years = float(age)
    except ValueError:
        return None
    for lo, hi in AGE_BANDS:
        if lo <= years <= hi:
            return band_label(lo, hi)
    return None


def participant_groups(participants: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """pid, age band and sex of each night, indexed by night (participant_id)."""
    p = participants.set_index("participant_id")
    return p.pid, p.age.map(age_band), p.sex.where(p.sex.isin(["F", "M"]))


def night_missing_rates(df: pd.DataFrame) -> pd.Series:
    """Per night, share of human 0-4 epochs where the headband gives -2."""
    human = pd.Series(is_stage(df.psg_hum.to_numpy()), index=df.index)
    g = df.assign(human=human, missing=human & (df.hb_ai == AI_MISSING)).groupby("sub", sort=False)
    return g.missing.sum() / g.human.sum()


def run(table: pd.DataFrame, participants: pd.DataFrame) -> dict:
    df = annotate(table)
    rates = night_rates(df)
    pid_of, band_of, sex_of = participant_groups(participants)
    stages = by_true_stage(df)
    return {
        "reference": "psg_hum", "candidate": "hb_ai", "n_nights": int(df["sub"].nunique()),
        "rate_definition": "1 - accuracy over epochs where human and headband both give a stage 0-4; "
                           "human 8 excluded; headband -2 reported as headband_missing_rate",
        "min_night_epochs": MIN_NIGHT_EPOCHS,
        "overall": group_summary(df),
        **stages,
        "by_distance_to_human_transition": by_distance_to_transition(df),
        "by_stage_and_distance_to_transition": stage_by_distance(df),
        "by_headband_artefact_flag": by_artefact_distance(df),
        "by_hour_of_night": {hour_label(h): group_summary(g) for h, g in df.groupby("hour")},
        "per_night_rate": rates.to_dict(),
        "per_night_headband_missing_rate": night_missing_rates(df).to_dict(),
        "per_participant": {"n_pids": int(pid_of.reindex(rates.dropna().index).nunique()),
                            "rate_summary": summarise(per_person(rates, pid_of)),
                            "rate_by_pid": per_person(rates, pid_of).to_dict()},
        "by_age_band": person_groups(rates, pid_of, band_of, AGE_BAND_ORDER),
        "n_nights_age_unknown": int(band_of.reindex(rates.index).isna().sum()),
        "by_sex": person_groups(rates, pid_of, sex_of, SEX_ORDER),
    }


def fig_distance(result: dict, out: Path):
    """Disagreement rate vs distance to the nearest human transition, one line per true stage."""
    grid = result["by_stage_and_distance_to_transition"]
    labels = [distance_label(k) for k in range(MAX_K + 2)]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for stage, cells in grid.items():
        x = [i for i, k in enumerate(labels) if cells[k]["n_epochs"] >= MIN_PLOT_EPOCHS]
        y = [100 * cells[labels[i]]["disagreement_rate"] for i in x]
        ax.plot(x, y, color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage], lw=1.6, ms=6.5,
                mfc=STAGE_COLORS[stage], mec=SURFACE, mew=1.2, label=stage)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("Epochs from the nearest human stage transition (0 = the transition epoch itself)")
    ax.set_ylabel("Disagreement rate, human vs headband (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(title="Human stage", frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              labelcolor=INK2, title_fontproperties={"weight": "normal"})
    ax.set_title("W, N2 and REM disagree most at transitions; N1 and N3 stay high at every distance")
    fig.text(0.01, 0.005, f"Epochs where both give a stage 0-4; points with fewer than {MIN_PLOT_EPOCHS} "
             "epochs are not drawn. Values: results/hotspots.json", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result = run(load_all(DATA), read_participants(DATA))
    fig_distance(result, FIG_DIR / "fig5_disagreement_vs_transition_distance.png")
    RESULTS.write_text(json.dumps(jsonable(result), indent=1))
    print(f"wrote {RESULTS} and {FIG_DIR}")


if __name__ == "__main__":
    main()
