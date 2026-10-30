"""Person effects: is headband trouble a property of the person, the night, or the person's habits? (D15)

Exploratory, label-only. Definitions: docs/definitions.md, "Person effects". Reads results/hotspots.json and
results/clinical.json plus participants.tsv, writes results/person_effects.json and figures/person_effects/*.png.

Run from the repo root (after boas_unc.hotspots and boas_unc.clinical):  uv run python -m boas_unc.person_effects
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from boas_unc.agreement import jsonable, summarise
from boas_unc.inventory_figures import GRID, INK, INK2, MUTED, SURFACE, style
from boas_unc.load import read_participants

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
HOTSPOTS = ROOT / "results" / "hotspots.json"
CLINICAL = ROOT / "results" / "clinical.json"
RESULTS = ROOT / "results" / "person_effects.json"
FIG_DIR = ROOT / "figures" / "person_effects"

HIGH_DROPOUT = 0.20   # D3
N_BOOT = 2000
BOOT_SEED = 0
CHANGE_METRICS = ["disagreement_rate", "dropout_share", "hum_sleep_efficiency", "hum_sol", "hum_waso", "hum_pct_rem"]
OUTCOMES = ["dropout_share", "disagreement_rate"]
SEX_COLORS = {"F": "#eb6834", "M": "#2a78d6"}


def night_table(hotspots: dict, clinical: dict, participants: pd.DataFrame) -> pd.DataFrame:
    """One row per night: person, sex, age, BMI, order, and the outcome and sleep columns."""
    nights = pd.DataFrame({"disagreement_rate": pd.Series(hotspots["per_night_rate"]),
                           "dropout_share": pd.Series(hotspots["per_night_headband_missing_rate"])})
    per_night = pd.DataFrame(clinical["per_night_metrics"]).T
    for col in ("hum_sleep_efficiency", "hum_sol", "hum_waso", "hum_pct_rem"):
        nights[col] = pd.to_numeric(per_night[col], errors="coerce")
    p = participants.set_index("participant_id")
    nights["pid"] = p.pid.reindex(nights.index)
    nights["sex"] = p.sex.where(p.sex.isin(["F", "M"])).reindex(nights.index)
    nights["age"] = pd.to_numeric(p.age, errors="coerce").reindex(nights.index)
    nights["bmi"] = pd.to_numeric(p.bmi, errors="coerce").reindex(nights.index)
    nights["order"] = [int(s.split("-")[1]) for s in nights.index]
    return nights.astype({c: float for c in nights.columns if c not in ("pid", "sex")})


def repeated(nights: pd.DataFrame) -> pd.DataFrame:
    """The nights of people who have at least two."""
    return nights[nights.groupby("pid")["order"].transform("size") >= 2]


def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects intraclass correlation. NaN with fewer than two groups of two or more."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    df = df[df.groupby("g").v.transform("size") >= 2]
    n, a = len(df), df.g.nunique()
    if a < 2 or n == a:
        return np.nan
    sizes = df.groupby("g").v.size()
    means = df.groupby("g").v.mean()
    grand = df.v.mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (a - 1))
    msw = float(((df.v - df.g.map(means)) ** 2).sum() / (n - a))
    n0 = (n - float((sizes ** 2).sum()) / n) / (a - 1)
    denom = msb + (n0 - 1) * msw
    return (msb - msw) / denom if denom else np.nan


def bootstrap_icc(values: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list:
    """95 % percentile interval of the ICC, resampling people."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    people = df.g.unique()
    by_person = {g: df[df.g == g].v.to_numpy() for g in people}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(people, len(people), replace=True)
        v = np.concatenate([by_person[g] for g in pick])
        g = np.concatenate([np.full(len(by_person[p]), i) for i, p in enumerate(pick)])
        draws.append(icc1(v, g))
    return [float(x) for x in np.nanpercentile(draws, [2.5, 97.5])]


def pair_values(nights: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of nights of the same person, as two aligned arrays (earlier number first)."""
    x, y = [], []
    for _, g in nights.sort_values("order").groupby("pid"):
        for a, b in combinations(g[col].tolist(), 2):
            x.append(a), y.append(b)
    return np.array(x), np.array(y)


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman correlation with its p-value; NaN when fewer than three pairs or a constant input."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": int(len(x)), "rho": np.nan, "p": np.nan}
    r = stats.spearmanr(x, y)
    return {"n": int(len(x)), "rho": float(r.statistic), "p": float(r.pvalue)}


def high_dropout_nights(nights: pd.DataFrame, threshold: float = HIGH_DROPOUT) -> list:
    """Each night above the threshold, its person, and the person's other nights."""
    out = []
    for sub, row in nights[nights.dropout_share > threshold].iterrows():
        others = nights[(nights.pid == row.pid) & (nights.index != sub)]
        out.append({"night": sub, "pid": row.pid, "dropout_share": float(row.dropout_share),
                    "other_nights": {s: float(v) for s, v in others.dropout_share.items()}})
    return out


def person_a(nights: pd.DataFrame) -> dict:
    """Person or night: ICC, pair correlation, and the high-dropout nights with their siblings."""
    rep = repeated(nights)
    x, y = pair_values(rep, "dropout_share")
    counts = rep.groupby("pid").size()
    return {"n_people": int(counts.size), "n_nights": int(len(rep)),
            "people_by_number_of_nights": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
            "icc1_dropout_share": icc1(rep.dropout_share.to_numpy(), rep.pid.to_numpy()),
            "icc1_95_interval": bootstrap_icc(rep.dropout_share.to_numpy(), rep.pid.to_numpy()),
            "icc1_disagreement_rate": icc1(rep.disagreement_rate.to_numpy(), rep.pid.to_numpy()),
            "spearman_pairs_dropout_share": spearman(x, y),
            "high_dropout_nights": high_dropout_nights(nights)}


def first_vs_later(nights: pd.DataFrame, col: str) -> dict:
    """Later (mean of the other nights) minus first night, per person with both defined (order = sub number)."""
    diffs = []
    for _, g in repeated(nights).sort_values("order").groupby("pid"):
        first, later = g[col].iloc[0], g[col].iloc[1:].mean()
        if not (np.isnan(first) or np.isnan(later)):
            diffs.append(later - first)
    d = np.array(diffs)
    p = np.nan
    if len(d) >= 5 and np.any(d != 0):
        p = float(stats.wilcoxon(d).pvalue)
    return {"n_people": int(len(d)), "difference": summarise(pd.Series(d)), "n_later_higher": int((d > 0).sum()),
            "n_later_lower": int((d < 0).sum()), "n_same": int((d == 0).sum()), "wilcoxon_p": p}


def person_means(nights: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per person: the mean of each column over their nights, and their sex."""
    out = nights.groupby("pid")[cols].mean()
    out["sex"] = nights.groupby("pid").sex.first()
    out["n_nights"] = nights.groupby("pid").size()
    return out


def sex_comparison(people: pd.DataFrame, outcome: str) -> dict:
    """Median and IQR of a person-level outcome per sex, and a Mann-Whitney p-value."""
    groups = {s: people.loc[people.sex == s, outcome].dropna() for s in ("F", "M")}
    p = float(stats.mannwhitneyu(groups["F"], groups["M"]).pvalue) if all(len(g) >= 2 for g in groups.values()) else np.nan
    return {"groups": {s: {"n_people": int(len(g)), **summarise(g)} for s, g in groups.items()}, "mannwhitney_p": p}


def covariate_correlation(people: pd.DataFrame, outcome: str, covariate: str) -> dict:
    """Spearman correlation of a person-level outcome with a person-level covariate."""
    ok = people[[outcome, covariate]].dropna()
    return {**spearman(ok[outcome].to_numpy(), ok[covariate].to_numpy()),
            "n_people_without_value": int(len(people) - len(ok))}


def person_c(nights: pd.DataFrame) -> dict:
    """Sex, BMI and age against the person-level outcomes."""
    people = person_means(nights, OUTCOMES + ["age", "bmi"])
    out = {"n_people": int(len(people))}
    for outcome in OUTCOMES:
        out[outcome] = {"sex": sex_comparison(people, outcome),
                        "bmi": covariate_correlation(people, outcome, "bmi"),
                        "age": covariate_correlation(people, outcome, "age")}
    return out


def run(nights: pd.DataFrame) -> dict:
    return {"order_assumption": "a person's nights are ordered by the number in sub-N (no dates in the label files)",
            "note": "exploratory; p-values are for orientation only",
            "a_person_or_night": person_a(nights),
            "b_first_vs_later": {m: first_vs_later(nights, m) for m in CHANGE_METRICS},
            "c_sex_bmi_age": person_c(nights)}


def fig_person_effects(nights: pd.DataFrame, result: dict, out: Path):
    """Three panels: one line per repeated-night person, first-vs-later efficiency, dropout against BMI."""
    rep = repeated(nights).sort_values("order")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    for _, g in rep.groupby("pid"):
        ax.plot(range(1, len(g) + 1), 100 * g.dropout_share, color=MUTED, lw=1.0, alpha=0.6, marker="o", ms=4,
                mec=SURFACE, mew=0.8)
    a = result["a_person_or_night"]
    ax.set_xticks([1, 2, 3], ["1st", "2nd", "3rd"])
    ax.set_xlabel("Night of the same person (ordered by sub number)")
    ax.set_ylabel("Headband dropout (%)")
    ax.set_title(f"Dropout by night, {a['n_people']} people with several nights", fontsize=10)
    ax.text(0.97, 0.95, f"ICC {a['icc1_dropout_share']:.2f}\n95 % interval {a['icc1_95_interval'][0]:.2f} to "
            f"{a['icc1_95_interval'][1]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax = axes[1]
    for _, g in rep.groupby("pid"):
        v = g.hum_sleep_efficiency.to_numpy()
        ax.plot([1, 2], [v[0], np.nanmean(v[1:])], color=MUTED, lw=1.0, alpha=0.6, marker="o", ms=4, mec=SURFACE, mew=0.8)
    f = result["b_first_vs_later"]["hum_sleep_efficiency"]
    ax.set_xticks([1, 2], ["First night", "Later nights (mean)"])
    ax.set_xlim(0.7, 2.3)
    ax.set_ylabel("Sleep efficiency, human scoring (%)")
    ax.set_title("Does sleep change after the first night?", fontsize=10)
    ax.text(0.03, 0.05, f"{f['n_later_higher']} up, {f['n_later_lower']} down of {f['n_people']} people",
            transform=ax.transAxes, fontsize=8, color=INK2)
    ax = axes[2]
    people = person_means(nights, ["dropout_share", "bmi"])
    for sex, g in people.groupby("sex"):
        ax.scatter(g.bmi, 100 * g.dropout_share, s=26, color=SEX_COLORS[sex], alpha=0.75, edgecolors="none",
                   label=f"{sex} ({len(g)} people)")
    c = result["c_sex_bmi_age"]["dropout_share"]["bmi"]
    ax.set_xlabel("BMI (mean over the person's nights)")
    ax.set_ylabel("Headband dropout (%)")
    ax.set_title("Dropout against BMI, one dot per person", fontsize=10)
    ax.text(0.97, 0.95, f"Spearman {c['rho']:.2f}, {c['n']} people", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
    for ax in axes:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Is headband trouble about the person, the night, or the setting?", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.text(0.01, 0.005, "Exploratory, few people. The order of nights is assumed from the sub number (the label files "
             "have no dates). Values: results/person_effects.json", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nights = night_table(json.loads(HOTSPOTS.read_text()), json.loads(CLINICAL.read_text()), read_participants(DATA))
    result = jsonable(run(nights))
    RESULTS.write_text(json.dumps(result, indent=1))
    fig_person_effects(nights, result, FIG_DIR / "fig15_person_effects.png")
    print(f"wrote {RESULTS} and {FIG_DIR}")


if __name__ == "__main__":
    main()
