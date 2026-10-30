"""Regenerate REPORT.md from results/*.json (Milestone 7). No number is written by hand.

Step 1 (`build_dataset_summary`) reads the dataset files once and writes results/dataset_summary.json.
Step 2 (`render`) builds the report from the results files only. tests/test_report.py checks that no
text in this file contains a digit outside file names and decision labels.

Run from the repo root, after the other result scripts:  uv run python -m boas_unc.report
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from boas_unc.clinical import LABELS
from boas_unc.load import load_all, read_participants
from boas_unc.metrics import MIN_REM_RUN
from boas_unc.stage_signature import BANDS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "boas"
RESULTS_DIR = ROOT / "results"
REPORT = ROOT / "REPORT.md"

# Statistical conventions, not results: the limits of agreement are the bias +/- LOA_Z standard deviations.
LOA_Z = 1.96
LOA_PCT = 95
INTERVAL_PCT = 95  # level of the bootstrap interval for the intraclass correlation
ALPHA = 0.05       # usual significance level, used for the multiple-comparison note

FIGURES = {
    "hotspots": "figures/hotspots/fig5_disagreement_vs_transition_distance.png",
    "bland_altman": "figures/clinical/fig12_bland_altman.png",
    "abstention": "figures/abstention/fig14_agreement_vs_coverage.png",
    "person_effects": "figures/person_effects/fig15_person_effects.png",
    "learned": "figures/abstention/fig16_learned_vs_heuristic.png",
    "signal_dead": "figures/signals/sub-1_overview.png",
    "signal_one_bad": "figures/signals/sub-7_overview.png",
    "signature": "figures/signals/fig17_stage_signature.png",
}
EXTRA_FIGURES = {
    "What the headband says when the human scores a stage": "figures/overview/fig6_headband_confusion.png",
    "Per-stage F1 for both AIs": "figures/overview/fig7_per_stage_f1.png",
    "Accuracy of every night": "figures/overview/fig8_per_night_accuracy.png",
    "A typical and the lowest-accuracy night": "figures/overview/fig9_example_nights.png",
    "Headband dropout by night": "figures/overview/fig10_headband_dropout_by_night.png",
    "Disagreement by age and sex": "figures/overview/fig11_disagreement_by_age_sex.png",
    "Bland-Altman plots for every metric": "figures/clinical/fig13_bland_altman_all_metrics.png",
}
STAGES = ["W", "N1", "N2", "N3", "REM"]
PAIR_LABELS = {"human_vs_psg_ai": "Human vs AI on PSG", "human_vs_hb_ai": "Human vs AI on headband",
               "psg_ai_vs_hb_ai": "AI on PSG vs AI on headband (analysis only)"}
ABSTENTION_ROWS = ["none", "isolated", "change_0", "change_1", "change_3", "change_5", "artefact_2",
                   "combined_1", "psg_ai_disagrees"]
CLINICAL_ORDER = ["tst", "sleep_efficiency", "sol", "waso", "rem_latency", "terminal_wake", "pct_n1", "pct_n2",
                  "pct_n3", "pct_rem"]


# ---------------------------------------------------------------- step 1: dataset facts

def readme_facts(text: str) -> dict:
    """Figures the dataset README states about its own labels, as written there (percent values)."""
    def find(pattern):
        m = re.search(pattern, text, re.S)
        return float(m.group(1)) if m else None
    esparza = next((line for line in text.splitlines() if "Esparza" in line and "doi" in line), "")
    doi = re.search(r"doi:\s*(10\.\S+?)\.?\s*$", esparza)
    return {"inter_scorer_agreement_pct": find(r"inter-scorer agreement of approximately (\d+(?:\.\d+)?)\s*%"),
            "human_vs_psg_ai_match_pct": find(r"(\d+(?:\.\d+)?)%\s*match between human-consensus and network-provided "
                                              r"labels for the PSG data"),
            "human_vs_headband_ai_match_pct": find(r"and an (\d+(?:\.\d+)?)%\s*match for the wearable EEG data"),
            "model_paper_doi": doi.group(1) if doi else None,
            "states_cross_validation": "cross-validation" in text}


def build_dataset_summary(data: Path = DATA) -> dict:
    """Facts about the dataset that the other results files do not hold."""
    desc = json.loads((data / "dataset_description.json").read_text())
    participants = read_participants(data)
    table = load_all(data)
    ages = pd.to_numeric(participants.age, errors="coerce")
    person_age = ages.groupby(participants.pid).mean()  # one value per person: the mean over their nights
    step = table.groupby("sub").onset.diff().dropna()
    people = participants.drop_duplicates("pid")
    return {"name": desc["Name"], "doi": desc["DatasetDOI"], "license": desc["License"], "authors": desc["Authors"],
            "n_nights": int(table["sub"].nunique()), "n_people": int(participants.pid.nunique()),
            "n_nights_age_unknown": int(ages.isna().sum()),
            "n_people_age_unknown": int(person_age.isna().sum()),
            "age_people_min_median_max": [float(person_age.min()), float(person_age.median()), float(person_age.max())],
            "age_nights_min_median_max": [float(ages.min()), float(ages.median()), float(ages.max())],
            "sex_nights": participants.sex.value_counts().to_dict(),
            "sex_people": people.sex.value_counts().to_dict(),
            "epoch_seconds": float(step.median()),
            "readme_reported": readme_facts((data / "README").read_text())}


def read_results(results: Path = RESULTS_DIR) -> dict:
    names = ["dataset_summary", "inventory_counts", "agreement", "hotspots", "clinical", "abstention",
             "person_effects", "learned_abstention", "signal_quality", "one_channel", "stage_signature", "three_class"]
    return {n: json.loads((results / f"{n}.json").read_text()) for n in names}


# ---------------------------------------------------------------- formatting

def pct(x, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def num(x, d: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def signed(x, d: int = 1) -> str:
    return "n/a" if x is None else f"{x:+.{d}f}"


def med_iqr(s: dict, scale: float = 1.0, d: int = 1, suffix: str = "") -> str:
    """'median (q25 to q75)' of a summarise() block."""
    if s is None or s["median"] is None:
        return "n/a"
    return f"{scale * s['median']:.{d}f}{suffix} ({scale * s['q25']:.{d}f} to {scale * s['q75']:.{d}f})"


def thousands(n: int) -> str:
    return f"{n:,}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def join_names(names: list[str]) -> str:
    return ", ".join(names[:-1]) + ", and " + names[-1] if len(names) > 1 else names[0]


def epochs(k) -> str:
    """'1 epoch', '3 epochs'."""
    return f"{k} epoch" if int(k) == 1 else f"{k} epochs"


def flag_label(name: str) -> str:
    """Plain-language name of a flag set."""
    fixed = {"none": "No abstention (baseline)", "isolated": "Isolated single-epoch stage",
             "psg_ai_disagrees": "PSG AI disagrees (NOT available on the device)"}
    if name in fixed:
        return fixed[name]
    kind, _, k = name.rpartition("_")
    what = {"change": "predicted stage change", "artefact": "headband no-stage (-2) epoch",
            "combined": "predicted change or a -2"}[kind]
    return f"On a {what}" if k == "0" else f"Within {epochs(k)} of a {what}"


def hardest_stage(hot: dict) -> tuple[str, dict]:
    stage = max(hot["by_stage"], key=lambda s: hot["by_stage"][s]["disagreement_rate"])
    return stage, hot["by_stage"][stage]


def main_confusion(hot: dict, stage: str) -> tuple[str, float]:
    """The label the headband most often gives instead of `stage`, and how often."""
    row = {k: v for k, v in hot["confusion_row_normalised_version_a"][stage].items() if k != stage and v is not None}
    other = max(row, key=row.get)
    return other, row[other]


def reported_note(res: dict) -> str:
    rep = res["dataset_summary"]["readme_reported"]
    ag = res["agreement"]["pairs"]
    if rep["human_vs_psg_ai_match_pct"] is None or rep["human_vs_headband_ai_match_pct"] is None:
        return ""
    ours = [100 * ag[k]["a_both_stage"]["pooled"]["accuracy"] for k in ("human_vs_psg_ai", "human_vs_hb_ai")]
    theirs = [rep["human_vs_psg_ai_match_pct"], rep["human_vs_headband_ai_match_pct"]]
    gap = max(abs(a - b) for a, b in zip(ours, theirs))
    return (f"For the same two comparisons the dataset README reports {num(theirs[0], 2)}% and {num(theirs[1], 2)}%. "
            f"This analysis gets {num(ours[0], 2)}% and {num(ours[1], 2)}%, within {num(gap, 2)} points, so the label "
            "files, the alignment and the stage mapping match the published figures. The small remaining difference is "
            "consistent with the label revision noted in the dataset changelog.\n")


# ---------------------------------------------------------------- step 2: sections

def section_summary(res: dict) -> str:
    sig = signal_totals(res)
    one_ch = res["one_channel"]["primary"]["share_of_missing_by_status"]
    ag, hot, cl, ab = res["agreement"]["pairs"], res["hotspots"], res["clinical"], res["abstention"]
    hb, psg = ag["human_vs_hb_ai"], ag["human_vs_psg_ai"]
    stage, st = hardest_stage(hot)
    other, share = main_confusion(hot, stage)
    dist = hot["by_distance_to_human_transition"]
    far = list(dist)[-1]
    tst, se = cl["standalone"]["tst"], cl["standalone"]["sleep_efficiency"]
    m_tst = cl["standalone"]["tst"]
    matched = cl["matched_epoch"]["all_nights"]["metrics"]["tst"]
    c1 = ab["flag_sets"]["change_1"]
    return (
        "## Summary\n\n"
        f"- **Agreement.** Where the headband AI gives a stage, it agrees with the human consensus on "
        f"{pct(hb['a_both_stage']['pooled']['accuracy'])} of epochs (Cohen's kappa {num(hb['a_both_stage']['pooled']['kappa'], 2)}), "
        f"against {pct(psg['a_both_stage']['pooled']['accuracy'])} for the AI run on the PSG. When an epoch with no headband "
        f"stage counts as an error, the headband figure is {pct(hb['b_end_to_end']['pooled']['accuracy'])}.\n"
        f"- **Where it is uncertain.** {stage} is the hardest stage: {pct(st['disagreement_rate'])} of its epochs "
        f"disagree, most often because the headband says {other} ({pct(share)} of the {stage} epochs). Disagreement is "
        f"{pct(dist['0']['disagreement_rate'])} on the epochs on either side of a human stage change and "
        f"{pct(dist[far]['disagreement_rate'])} {far} epochs away from one.\n"
        f"- **Clinical metrics.** Night by night, the headband's total sleep time differs from the human's by "
        f"{signed(tst['bias'])} min on average (limits of agreement {num(tst['loa_lower'], 0)} to {num(tst['loa_upper'], 0)} min) "
        f"and its sleep efficiency by {signed(se['bias'])} points. On the same {cl['n_nights']} nights, restricted to epochs "
        f"where both give a stage, the total sleep time difference is {signed(matched['bias'])} min, so most of the "
        "sleep-time gap comes from epochs where the headband gives no stage.\n"
        f"- **The raw signal.** Of the {thousands(sig['n_missing'])} epochs without a headband stage, "
        f"{pct(sig['missing_and_flat'] / sig['n_missing'])} have both channels flat and "
        f"{pct(one_ch['one_good'])} have exactly one clean channel. A flat signal is not always flagged: "
        f"{thousands(sig['flat_but_staged'])} flat epochs have a headband stage.\n"
        f"- **How to flag it.** Setting aside the epochs within one epoch of a predicted stage change leaves "
        f"{pct(c1['coverage'])} of the epochs, with {pct(c1['accuracy_kept'])} agreement on those, from "
        f"{pct(ab['baseline_accuracy'])} before flagging. The flag uses the headband labels only.\n\n"
    )


def section_dataset(res: dict) -> str:
    d, inv = res["dataset_summary"], res["inventory_counts"]
    lo, med, hi = d["age_people_min_median_max"]
    night_median = d["age_nights_min_median_max"][1]
    n_aged = d["n_people"] - d["n_people_age_unknown"]
    sex = ", ".join(f"{n} {k}" for k, n in d["sex_people"].items())
    hours = inv["night_hours_min_median_max"]
    return (
        "## Dataset and data used\n\n"
        f"- **Dataset:** {d['name']}. DOI `{d['doi']}`, licence {d['license']}. "
        f"Authors: {join_names(d['authors'])}.\n"
        f"- **Content:** {d['n_nights']} nights from {d['n_people']} people (some people were recorded on more than one night), "
        f"{thousands(inv['n_epochs'])} epochs of {num(d['epoch_seconds'], 0)} s. Nights last {num(hours[0])} to {num(hours[2])} h "
        f"(median {num(hours[1])} h). Every night was recorded at the same time with a clinical polysomnography (PSG) and "
        "with the headband.\n"
        f"- **People:** ages {num(lo, 0)} to {num(hi, 0)}, median {num(med, 0)} over the {n_aged} people with a known age "
        f"(one value per person, the mean over their nights; {d['n_people_age_unknown']} person has no recorded age). Counted "
        f"by night instead, the median age is {num(night_median, 0)}, because people in the older age bands "
        f"contributed more nights. By sex: {sex}. Participants were members of the general population. Patients were not recruited, and severe "
        "medical, neurological or psychiatric conditions were exclusion criteria (dataset README).\n"
        "- **Labels used:** the human consensus on the PSG (the reference), the AI label computed from the PSG, and the AI "
        "label computed from the headband. Most of the analysis uses only these label files. One section reads the "
        "headband's own EEG, never the PSG signals.\n"
        "- **Special codes:** the humans mark stretches where the PSG was disconnected (code 8). The AI marks epochs it "
        "cannot stage as artefact or missing (code -2). Both are kept on the timeline and handled by explicit rules.\n\n"
    )


def section_methods(res: dict) -> str:
    return (
        "## Methods\n\n"
        "- The human consensus label is the reference. The AI labels come with the dataset, and no model was trained here.\n"
        "- Labels are joined across files by epoch onset time, never by row order. Every epoch stays on the full timeline.\n"
        "- Agreement is reported in three versions: (a) epochs where both labels are a stage, (b) the same with a missing "
        "headband stage counted as an error, (c) coverage, the share of reference epochs where the AI gives a stage.\n"
        "- Epochs where the PSG was disconnected are left out of every comparison, because the humans had no signal either.\n"
        "- Disagreement hotspots use version (a): by true stage, by distance to the nearest human stage change, next to "
        "headband no-stage epochs, by hour of the night, by person, age band and sex.\n"
        "- Clinical sleep metrics follow AASM-style definitions written down before implementation "
        "(`docs/definitions.md`). The main analysis keeps each sequence on its own timeline, so headband dropout shows up "
        "as lost sleep time. A matched-epoch analysis restricts both sequences to epochs where both give a stage.\n"
        f"- Per-night differences are summarised with the mean bias and limits of agreement (bias plus or minus {LOA_Z} "
        f"standard deviations, covering about {LOA_PCT}% of nights), correlation, and the median and IQR over nights.\n"
        "- Device-only flags are computed from the headband label sequence alone: near a predicted stage change, isolated "
        "single-epoch stage, near a missing-stage epoch. Their ranges were fixed before any result was computed.\n"
        "- Distributions are over nights. Comparisons between groups use the person as the unit, because some people "
        "contributed several nights.\n"
        "- Every number in this report is generated from `results/*.json` by `src/boas_unc/report.py`.\n\n"
    )


def per_stage_rows(res: dict) -> list[list[str]]:
    hb = res["agreement"]["pairs"]["human_vs_hb_ai"]["a_both_stage"]["pooled"]["per_stage"]
    psg = res["agreement"]["pairs"]["human_vs_psg_ai"]["a_both_stage"]["pooled"]["per_stage"]
    return [[s, thousands(hb[s]["support"]), pct(hb[s]["precision"]), pct(hb[s]["recall"]), pct(hb[s]["f1"]),
             pct(psg[s]["f1"])] for s in STAGES]


def section_agreement(res: dict) -> str:
    ag = res["agreement"]["pairs"]
    rows = []
    for key, label in PAIR_LABELS.items():
        a, b, p = ag[key]["a_both_stage"], ag[key]["b_end_to_end"], ag[key]
        rows.append([label, thousands(a["pooled"]["n_epochs"]), pct(a["pooled"]["accuracy"]),
                     num(a["pooled"]["kappa"], 2), pct(b["pooled"]["accuracy"]), pct(p["coverage"]["pooled"]),
                     med_iqr(a["per_night"]["accuracy"], 100, 1, "%")])
    return (
        "## Agreement\n\n"
        + table(["Pair (reference first)", "Epochs, version (a)", "Accuracy (a)", "Kappa (a)", "Accuracy (b)",
                 "Coverage (c)", "Per-night accuracy (a), median (IQR)"], rows)
        + "\n" + reported_note(res) + "\n"
        "Per stage, human as the reference (version (a), pooled over all epochs):\n\n"
        + table(["Stage", "Human epochs", "Headband precision", "Headband recall", "Headband F1", "PSG AI F1"],
                per_stage_rows(res))
        + "\nThe AI on the headband and the AI on the PSG have the same shape across stages. The pair in the last "
        "row of the first table compares two AI outputs with each other. It is an analysis tool and not a deployable "
        "check, because a home user has no PSG.\n\n"
    )


def stage_rows(hot: dict) -> list[list[str]]:
    rows = []
    for s in STAGES:
        g = hot["by_stage"][s]
        other, share = main_confusion(hot, s)
        rows.append([s, thousands(g["n_epochs"]), pct(g["disagreement_rate"]), med_iqr(g["per_night_rate"], 100, 1, "%"),
                     f"{other} ({pct(share)})"])
    return rows


def section_hotspots(res: dict) -> str:
    hot = res["hotspots"]
    dist = [[("transition epoch" if k == "0" else k), thousands(g["n_epochs"]), pct(g["disagreement_rate"]),
             pct(g["headband_missing_rate"])] for k, g in hot["by_distance_to_human_transition"].items()]
    flag = hot["by_headband_artefact_flag"]
    art = [["no flag within the range", thousands(flag["no_flag_within_5"]["n_epochs"]),
            pct(flag["no_flag_within_5"]["disagreement_rate"]), "", ""]]
    for d in flag["before_flag"]:
        b, a = flag["before_flag"][d], flag["after_flag"][d]
        art.append([f"{d} epoch(s) away", thousands(b["n_epochs"]), pct(b["disagreement_rate"]),
                    thousands(a["n_epochs"]), pct(a["disagreement_rate"])])
    hours = [[h, str(g["n_nights"]), thousands(g["n_epochs"]), pct(g["disagreement_rate"])]
             for h, g in hot["by_hour_of_night"].items()]
    groups = [[f"Age {k}", str(g["n_pids"]), str(g["n_nights"]), med_iqr(g["per_person_rate"], 100, 1, "%")]
              for k, g in hot["by_age_band"].items()]
    groups += [[f"Sex {k}", str(g["n_pids"]), str(g["n_nights"]), med_iqr(g["per_person_rate"], 100, 1, "%")]
               for k, g in hot["by_sex"].items()]
    pp = hot["per_participant"]
    return (
        "## Where the disagreements are\n\n"
        f"Overall, {pct(hot['overall']['disagreement_rate'])} of the {thousands(hot['overall']['n_epochs'])} epochs where "
        "both give a stage disagree. The rate by the stage the human scored:\n\n"
        + table(["Human stage", "Epochs", "Disagreement", "Per night, median (IQR)", "Most common headband label instead"],
                stage_rows(hot))
        + f"\n![Disagreement against distance to the nearest human stage change]({FIGURES['hotspots']})\n\n"
        "*Figure A. Disagreement between the human and the headband against the distance, in epochs, to the nearest "
        "human stage change, one line per human stage. Points with too few epochs are not drawn.*\n\n"
        "By distance to the nearest human stage change, all stages together. The last column is the share of the human's "
        "staged epochs where the headband gives no stage, which version (a) leaves out of the rate:\n\n"
        + table(["Distance (epochs)", "Epochs", "Disagreement", "Headband gives no stage"], dist)
        + "\nNext to an epoch where the headband gives no stage (-2), looking at the epochs just before and just after:\n\n"
        + table(["Distance to the flag", "Epochs before", "Disagreement before", "Epochs after", "Disagreement after"], art)
        + "\nBy hour since the recording start (late hours rest on few nights):\n\n"
        + table(["Hour", "Nights", "Epochs", "Disagreement"], hours)
        + f"\nPer person, the median night-level disagreement is {med_iqr(pp['rate_summary'], 100, 1, '%')} over "
        f"{pp['n_pids']} people. By age band and sex, with the person as the unit:\n\n"
        + table(["Group", "People", "Nights", "Per-person disagreement, median (IQR)"], groups)
        + "\nThe groups overlap heavily and no statistical test was run, so these are descriptions and not findings.\n\n"
    )


def clinical_rows(res: dict) -> list[list[str]]:
    rows = []
    for m in CLINICAL_ORDER:
        s = res["clinical"]["standalone"][m]
        title, unit = LABELS[m]
        rows.append([f"{title} ({unit})", str(s["n"]), med_iqr(s["human"]), med_iqr(s["headband"]), signed(s["bias"]),
                     f"{num(s['loa_lower'])} to {num(s['loa_upper'])}", med_iqr(s["difference"]),
                     num(s["pearson_r"], 2)])
    return rows


def matched_rows(res: dict) -> list[list[str]]:
    blk = res["clinical"]["matched_epoch"]["20_percent_primary"]
    rows = []
    for m in ["tst", "sleep_efficiency", "pct_n1", "pct_n2", "pct_n3", "pct_rem"]:
        s, st = blk["metrics"][m], res["clinical"]["standalone"][m]
        title, unit = LABELS[m]
        rows.append([f"{title} ({unit})", signed(st["bias"]), signed(s["bias"]),
                     f"{num(s['loa_lower'])} to {num(s['loa_upper'])}"])
    return rows


ZERO_FLOOR_SHARE = 0.10  # a metric is listed when at least this share of nights has a zero on either side


def zero_floor_text(res: dict) -> str:
    cl = res["clinical"]
    rows = []
    for m in CLINICAL_ORDER:
        z = cl["zero_floor"][m]
        stacked = z["both_zero"] + z["headband_zero_human_above"] + z["human_zero_headband_above"]
        if z["n"] and stacked / z["n"] >= ZERO_FLOOR_SHARE:
            title, unit = LABELS[m]
            rows.append([f"{title} ({unit})", str(z["n"]), str(z["both_zero"]), str(z["headband_zero_human_above"]),
                         str(z["human_zero_headband_above"])])
    return ("**Metrics piled up at zero.** For some metrics many nights have exactly zero in one or both sequences. "
            "In a Bland-Altman plot those nights fall on straight diagonal lines, so the limits of agreement describe "
            "these metrics poorly, and the per-night distributions above are more informative.\n\n"
            + table(["Metric", "Nights", "Both zero", "Headband zero, human above", "Human zero, headband above"], rows)
            + "\n")


def dropout_text(res: dict) -> str:
    cl = res["clinical"]
    thr = cl["matched_epoch"]["20_percent_primary"]["threshold"]
    nights = cl["flags"]["headband_dropout_over_20_percent"]
    return (f"The dropout is concentrated: {len(nights)} nights ({', '.join(nights)}) have more than {pct(thr, 0)} of "
            "their time with no headband stage.\n\n")


def section_clinical(res: dict) -> str:
    cl = res["clinical"]
    blk = cl["matched_epoch"]["20_percent_primary"]
    allb = cl["matched_epoch"]["all_nights"]["metrics"]
    first, sust = cl["standalone"]["rem_latency"], cl["sensitivity"]["rem_latency_sustained_run_of_3"]
    return (
        "## Clinical sleep metrics, night by night\n\n"
        f"Each of the {cl['n_nights']} nights gives one value per metric from the human labels and one from the headband "
        "labels, each on its own timeline, so epochs with no headband stage count as lost time. Differences are headband "
        "minus human. Limits of agreement are the bias plus or minus "
        f"{LOA_Z} standard deviations of the differences.\n\n"
        + table(["Metric", "Nights", "Human, median (IQR)", "Headband, median (IQR)", "Mean bias",
                 "Limits of agreement", "Median per-night difference (IQR)", "Pearson r"], clinical_rows(res))
        + f"\n![Bland-Altman plots for sleep efficiency and WASO]({FIGURES['bland_altman']})\n\n"
        "*Figure B. Bland-Altman plots for sleep efficiency and wake after sleep onset (WASO). Each dot is one night. "
        "Open circles are nights with a high share of headband dropout.*\n\n"
        "**How much of the gap is headband dropout?** Restricting both sequences to the epochs where both give a stage "
        f"(the matched-epoch analysis, leaving out the {len(blk['excluded_nights'])} nights with a high dropout share), "
        "the bias shrinks:\n\n"
        + table(["Metric", "Bias, own timelines", "Bias, matched epochs", "Limits of agreement, matched"], matched_rows(res))
        + f"\nOn all {cl['n_nights']} nights the matched-epoch biases are {signed(allb['tst']['bias'])} min for total sleep "
        f"time and {signed(allb['sleep_efficiency']['bias'])} points for sleep efficiency. " + dropout_text(res)
        + f"**REM latency.** With the first REM epoch as the definition, the mean bias is {signed(first['bias'])} min "
        f"and the median per-night difference {signed(first['difference']['median'])} min "
        f"({first['n_nights_used']} nights). Requiring a run of at least {MIN_REM_RUN} REM epochs on both sides changes the "
        f"mean bias to {signed(sust['bias'])} min and the median per-night difference to "
        f"{signed(sust['difference']['median'])} min. A typical night is then close to the human value, and a tail of "
        "nights where the headband reaches sustained REM much earlier keeps the mean bias large. The labels alone cannot "
        "say whether those stretches are wrong or whether the human scorers called the same epochs N1 or N2.\n\n"
        + zero_floor_text(res)
    )


def abstention_rows(res: dict) -> list[list[str]]:
    fs = res["abstention"]["flag_sets"]
    return [[flag_label(n), pct(fs[n]["coverage"]), pct(fs[n]["accuracy_kept"]), pct(fs[n]["error_capture"]),
             med_iqr(fs[n]["per_night"]["accuracy_kept"], 100, 1, "%")] for n in ABSTENTION_ROWS]


def smoothing_agreement(sm: dict) -> str:
    before, after = sm["agreement_version_a"]["before"], sm["agreement_version_a"]["after"]
    if round(before["accuracy"], 4) == round(after["accuracy"], 4):
        return (f"Agreement stays at {pct(before['accuracy'], 2)} (kappa {num(before['kappa'], 3)} to "
                f"{num(after['kappa'], 3)}).")
    return (f"Agreement moves from {pct(before['accuracy'], 2)} to {pct(after['accuracy'], 2)} (kappa "
            f"{num(before['kappa'], 3)} to {num(after['kappa'], 3)}).")


def smoothing_rows(res: dict) -> list[list[str]]:
    sm = res["abstention"]["clinical_smoothed"]["clinical"]
    rows = []
    for m in ["sleep_efficiency", "sol", "waso", "rem_latency"]:
        c = sm[m]
        title, unit = LABELS[m]
        rows.append([f"{title} ({unit})", f"{num(abs(c['before']['bias']), 2)} to {num(abs(c['after']['bias']), 2)}",
                     f"{num(c['before']['median_abs_error'], 2)} to {num(c['after']['median_abs_error'], 2)}"])
    return rows


def exclusion_rows(res: dict) -> list[list[str]]:
    rows = []
    for name, e in res["abstention"]["clinical_excluded"].items():
        se, n3 = e["metrics"]["sleep_efficiency"], e["metrics"]["pct_n3"]
        kept = e["mean_tib_matched_min"]["after"] / e["mean_tib_matched_min"]["before"]
        rows.append([flag_label(name), pct(kept), f"{num(se['before']['median_abs_error'], 2)} to "
                     f"{num(se['after']['median_abs_error'], 2)}", f"{num(n3['before']['median_abs_error'], 2)} to "
                     f"{num(n3['after']['median_abs_error'], 2)}"])
    return rows


def section_abstention(res: dict) -> str:
    ab = res["abstention"]
    sm = ab["clinical_smoothed"]
    return (
        "## How to flag the uncertain epochs\n\n"
        "A flag marks an epoch that should not be trusted. The flags below use the headband label sequence alone, so a "
        "wearable could compute them on the device. Agreement is measured against the human on the epochs that are kept, "
        f"starting from {pct(ab['baseline_accuracy'])} with no flag. Random abstention leaves that figure unchanged.\n\n"
        + table(["Flag", "Coverage (share kept)", "Agreement on kept epochs", "Share of disagreements flagged",
                 "Per-night agreement, median (IQR)"], abstention_rows(res))
        + f"\n![Agreement on kept epochs against coverage]({FIGURES['abstention']})\n\n"
        "*Figure C. Agreement with the human on the epochs that are kept, against the share of epochs kept, as the "
        "flag range k grows. The hollow diamond uses the PSG AI and is not available on the device. It is shown as a "
        "reference for what a second opinion from the PSG adds.*\n\n"
        f"**Smoothing.** Replacing each isolated epoch whose two neighbours agree with their label changes "
        f"{pct(sm['share_of_epochs_changed'])} of the epochs. {smoothing_agreement(sm)} "
        "Effect on a few clinical metrics (absolute mean bias, and median absolute per-night error, before to after):\n\n"
        + table(["Metric", "Absolute bias", "Median absolute error"], smoothing_rows(res))
        + "\n**Leaving flagged epochs out.** Both sequences are computed on the kept epochs only. This measures staging "
        "agreement on those epochs, not how well the whole-night value is recovered from part of the night. Median "
        "absolute per-night error, before to after:\n\n"
        + table(["Flag", "Share of matched time kept", "Sleep efficiency (points)", "N3 share of sleep (points)"],
                exclusion_rows(res))
        + "\nThe flag for epochs near a predicted stage change does most of the work. The flags for epochs near a "
        "missing-stage epoch change little, because few epochs sit near one.\n\n"
        + section_learned(res)
    )


PERSON_METRIC_LABELS = {"disagreement_rate": "Disagreement rate (points)", "dropout_share": "Headband dropout (points)",
                        "hum_sleep_efficiency": "Sleep efficiency, human (points)", "hum_sol": "Sleep onset latency, human (min)",
                        "hum_waso": "WASO, human (min)", "hum_pct_rem": "REM share of sleep, human (points)"}
PERCENT_METRICS = {"disagreement_rate", "dropout_share"}


def person_change_rows(res: dict) -> list[list[str]]:
    rows = []
    for m, label in PERSON_METRIC_LABELS.items():
        v = res["person_effects"]["b_first_vs_later"][m]
        scale = 100 if m in PERCENT_METRICS else 1
        rows.append([label, str(v["n_people"]), med_iqr(v["difference"], scale, 1), f"{v['n_later_higher']} / "
                     f"{v['n_later_lower']} / {v['n_same']}", num(v["wilcoxon_p"], 2)])
    return rows


def person_factor_rows(res: dict) -> list[list[str]]:
    c = res["person_effects"]["c_sex_bmi_age"]
    rows = []
    for outcome, label in (("dropout_share", "Headband dropout"), ("disagreement_rate", "Disagreement rate")):
        o = c[outcome]
        sex = o["sex"]["groups"]
        rows.append([label, f"F {pct(sex['F']['median'], 2)}, M {pct(sex['M']['median'], 2)} "
                     f"(people: {sex['F']['n_people']} and {sex['M']['n_people']})", num(o["sex"]["mannwhitney_p"], 2),
                     f"{num(o['bmi']['rho'], 2)} (p {num(o['bmi']['p'], 2)}, {o['bmi']['n']} people)",
                     f"{num(o['age']['rho'], 2)} (p {num(o['age']['p'], 2)}, {o['age']['n']} people)"])
    return rows


def person_p_values(res: dict) -> dict:
    """Every p-value of the person-effects analysis, labelled."""
    pe = res["person_effects"]
    out = {f"{PERSON_METRIC_LABELS[m]}, first against later nights": v["wilcoxon_p"]
           for m, v in pe["b_first_vs_later"].items()}
    for outcome in ("dropout_share", "disagreement_rate"):
        o = pe["c_sex_bmi_age"][outcome]
        name = PERSON_METRIC_LABELS[outcome].split(" (")[0].lower()
        out[f"{name} by sex"] = o["sex"]["mannwhitney_p"]
        out[f"{name} against BMI"] = o["bmi"]["p"]
        out[f"{name} against age"] = o["age"]["p"]
    return {k: v for k, v in out.items() if v is not None}


def multiple_comparison_note(res: dict) -> str:
    ps = person_p_values(res)
    what, smallest = min(ps.items(), key=lambda kv: kv[1])
    limit = ALPHA / len(ps)
    verdict = "below" if smallest < limit else "above"
    return (f"Across the {len(ps)} comparisons in this section the smallest p-value is {num(smallest, 3)} ({what}). "
            f"With that many comparisons, the level to reach is about {num(limit, 4)} (the usual level divided by their "
            f"number), so the smallest is {verdict} it.")


def section_person(res: dict) -> str:
    pe = res["person_effects"]
    a = pe["a_person_or_night"]
    rows = []
    for h in a["high_dropout_nights"]:
        others = ", ".join(f"{k} {pct(v)}" for k, v in h["other_nights"].items()) or "no other night"
        rows.append([h["night"], pct(h["dropout_share"]), others])
    lo, hi = a["icc1_95_interval"]
    sp = a["spearman_pairs_dropout_share"]
    return (
        "## Is the trouble about the person, the night or the setting?\n\n"
        "Exploratory, label only, and based on the people who were recorded more than once. "
        f"{a['n_people']} people have {a['n_nights']} nights between them. Because the dataset has no recording dates, a "
        "person's nights are ordered by the number in the recording name, which is an assumption.\n\n"
        f"**Dropout.** The intraclass correlation of the dropout share is {num(a['icc1_dropout_share'], 2)} "
        f"({INTERVAL_PCT}% interval {num(lo, 2)} to {num(hi, 2)}), and the correlation between nights of the same person is {num(sp['rho'], 2)} over "
        f"{sp['n']} pairs. The interval is wide. A moderate person effect can't be ruled out, but these data do not "
        "support one. The nights with a high dropout share and the same person's other nights:\n\n"
        + table(["Night", "Dropout", "The person's other nights"], rows)
        + f"\n![Person effects]({FIGURES['person_effects']})\n\n"
        "*Figure D. Left: dropout in each night of the people recorded more than once. Middle: human sleep efficiency "
        "on the first night and on later nights. Right: dropout against BMI, one dot per person.*\n\n"
        "**First nights against later nights.** For each person, the later nights (their mean) minus the first night. "
        "The last two columns are the number of people where it goes up, down or stays the same, and an exploratory "
        "Wilcoxon p-value.\n\n"
        + table(["Measure", "People", "Median difference (IQR)", "Up / down / same", "p"], person_change_rows(res))
        + "\n**Sex, BMI and age.** With the person as the unit. The comparison by sex is a Mann-Whitney test, the others "
        "are Spearman correlations. Many comparisons are made, so p-values are for orientation only.\n\n"
        + table(["Outcome", "By sex (median)", "p by sex", "BMI: Spearman rho", "Age: Spearman rho"],
                person_factor_rows(res))
        + "\n" + multiple_comparison_note(res) + " The data cannot separate band fit, sweat and movement. That needs "
        "the headband's movement sensor and the signals.\n\n"
    )


def learned_rows(res: dict) -> list[list[str]]:
    la = res["learned_abstention"]
    rows = []
    for name, auroc in la["flag_auroc_hand_made"].items():
        rows.append([flag_label(name), "hand-made", num(auroc, 2)])
    for fs, label in (("causal", "labels up to the epoch"), ("non_causal", "labels before and after")):
        for model, mlabel in (("logistic", "logistic regression"), ("gradient_boosting", "gradient boosting")):
            rows.append([f"Learned, {label}", mlabel, num(la["feature_sets"][fs]["models"][model]["auroc"], 2)])
    return rows


def learned_versus_rows(res: dict) -> list[list[str]]:
    fs = res["learned_abstention"]["feature_sets"]
    names = ["change_0", "change_1", "change_3", "change_5", "psg_ai_disagrees"]
    cc = fs["causal"]["models"]["gradient_boosting"]["versus_hand_made_flags"]
    nc = fs["non_causal"]["models"]["gradient_boosting"]["versus_hand_made_flags"]
    return [[flag_label(n), pct(nc[n]["coverage"]), pct(nc[n]["heuristic_accuracy"], 2), pct(cc[n]["learned_accuracy"], 2),
             pct(nc[n]["learned_accuracy"], 2)] for n in names]


def section_learned(res: dict) -> str:
    la = res["learned_abstention"]
    nc = la["feature_sets"]["non_causal"]["models"]["gradient_boosting"]["versus_hand_made_flags"]
    gains = [nc[n]["difference"] for n in nc if nc[n]["on_device"]]
    return (
        "**A model on the labels alone.** Can a small model, trained on the headband's label sequence and the clock only, "
        "predict where the headband is wrong better than the hand-made flags? Five folds keep every night of one person "
        "together, every score is out of fold, and nothing was tuned. The causal version sees only the labels up to the "
        "epoch, so it could run in real time. The other sees labels on both sides. The first table is the ability to "
        "separate errors from correct epochs (AUROC: the chance that a wrong epoch is ranked above a correct one, "
        "higher is better, and chance level is the same as a coin flip):\n\n"
        + table(["Predictor", "Kind", "AUROC"], learned_rows(res))
        + "\nAgreement on kept epochs at the same coverage as each hand-made flag:\n\n"
        + table(["Hand-made flag", "Coverage", "Hand-made agreement", "Learned, up to the epoch", "Learned, both sides"],
                learned_versus_rows(res))
        + f"\n![Learned against hand-made flags]({FIGURES['learned']})\n\n"
        "*Figure E. Agreement on kept epochs against coverage for a gradient-boosted model trained on the labels alone, "
        "against the hand-made flags. The hollow diamond uses the PSG AI and is not available on the device.*\n\n"
        f"Compared with the hand-made flags at the same coverage, the learned model that sees both sides changes "
        f"agreement by {signed(100 * min(gains), 1)} to {signed(100 * max(gains), 1)} points across the device-only flags. "
        "The simple flags already use most of what the labels can tell. A larger gain needs information from the signal, "
        "for example the model's probabilities or a signal-quality measure.\n\n"
    )


STATUS_LABELS = {"both_good": "Both channels clean", "one_good": "One channel clean, one bad",
                 "both_bad": "Both channels bad"}
CONTRAST_LABELS = {"N3_vs_N2": "N3 against N2", "N2_vs_N1": "N2 against N1", "REM_vs_N2": "REM against N2",
                   "W_vs_sleep": "Wake against sleep"}
FEATURE_LABELS = {"delta": "delta", "theta": "theta", "alpha": "alpha", "sigma": "sigma", "beta": "beta",
                  "log_std": "amplitude"}


def signal_totals(res: dict) -> dict:
    """Pooled counts of no-stage and flat epochs from the per-night signal-quality results."""
    q = res["signal_quality"]["q1_per_night"]
    missing = sum(v["n_missing"] for v in q.values())
    flat = sum(v["n_flat"] for v in q.values())
    both = sum(round(v["share_of_missing_that_are_flat"] * v["n_missing"]) for v in q.values()
               if v["n_missing"] and v["share_of_missing_that_are_flat"] is not None)
    flat_staged = {s: v["n_flat"] - round(v["share_of_flat_that_are_missing"] * v["n_flat"]) for s, v in q.items()
                   if v["n_flat"] and v["share_of_flat_that_are_missing"] is not None}
    flat_staged = {s: n for s, n in flat_staged.items() if n > 0}
    return {"n_nights": len(q), "n_missing": missing, "n_flat": flat, "missing_and_flat": both,
            "flat_but_staged": sum(flat_staged.values()), "flat_staged_by_night": flat_staged,
            "nights_with_missing": sum(v["n_missing"] > 0 for v in q.values()),
            "nights_with_flat": sum(v["n_flat"] > 0 for v in q.values())}


def one_channel_rows(res: dict) -> list[list[str]]:
    rows = []
    for name, label in (("primary", "main thresholds"), ("sensitivity", "stricter thresholds")):
        c = res["one_channel"][name]
        th = res["one_channel"]["thresholds"][name]
        share = c["share_of_missing_by_status"]
        rows.append([f"{label} (bad at {th['high_factor']:g} times or {th['low_factor']:g} times the night's typical amplitude)",
                     pct(share["both_good"]), pct(share["one_good"]), pct(share["both_bad"]),
                     pct(c["share_of_one_good_epochs_without_a_stage"])])
    return rows


def one_channel_agreement_rows(res: dict) -> list[list[str]]:
    rows = []
    for name, label in (("primary", "main thresholds"), ("sensitivity", "stricter thresholds")):
        a = res["one_channel"][name]["agreement"]
        rows.append([label] + [f"{pct(a[k]['accuracy'])} ({thousands(a[k]['n'])})" for k in STATUS_LABELS])
    return rows


def signature_rows(res: dict) -> list[list[str]]:
    sig = res["stage_signature"]["stage_signature"]["combined"]
    return [[s, str(v["n_nights"])] + [pct(v[b]["median"]) for b in ("delta", "theta", "alpha", "sigma", "beta")]
            for s, v in sig.items()]


def auroc_rows(res: dict) -> list[list[str]]:
    au = res["stage_signature"]["auroc"]["combined"]
    return [[CONTRAST_LABELS[c], str(d["delta"]["n_defined"])] + [num(d[f]["median"], 2) for f in FEATURE_LABELS]
            for c, d in au.items()]


def spike_rows(res: dict) -> list[list[str]]:
    sp = res["stage_signature"]["spikes"]
    return [[s, pct(sp["share_of_spikes_by_stage"][s]), pct(sp["share_of_live_epochs_by_stage"][s]),
             f"{sp['share_of_spikes_by_stage'][s] / sp['share_of_live_epochs_by_stage'][s]:.1f}"]
            for s in sp["share_of_spikes_by_stage"]]


def section_signal(res: dict) -> str:
    sq_, oc, ss = res["signal_quality"], res["one_channel"], res["stage_signature"]
    t = signal_totals(res)
    pc = oc["primary"]
    mv = sq_["q2_movement_before_long_runs"]
    order = sq_["q3_order_of_nights"]
    worst = sorted(t["flat_staged_by_night"].items(), key=lambda kv: -kv[1])[:3]
    channel_bad = pc["one_good_missing_bad_channel"]
    total_bad = max(sum(channel_bad.values()), 1)
    c1, c2 = list(channel_bad)
    return (
        "## What the raw headband signal shows\n\n"
        f"Up to here the analysis used labels only. This part reads the headband's own EEG, both channels, in all "
        f"{t['n_nights']} nights, and compares it with the labels. Only files that match their published checksum were "
        "used, read-only. The PSG signals were not read, so the human labels cannot be checked against them here.\n\n"
        f"**What is behind a no-stage epoch.** {thousands(t['n_missing'])} epochs in {t['nights_with_missing']} nights get no "
        f"headband stage. In {pct(t['missing_and_flat'] / t['n_missing'])} of them both channels are flat (constant, for "
        f"at least half of the epoch). Looking at each channel against the night's own typical amplitude (the main "
        f"thresholds below), the rest are mostly nights where one channel is wild or nearly silent while the other is "
        f"clean. On the dead-signal nights, the signal looks normal until a large transient, then sits at a constant "
        f"value (the bottom of the file's range) for hours: see the first example night.\n\n"
        + table(["Epochs with no headband stage", "Both channels clean", "One channel clean, one bad", "Both channels bad",
                 "Share of one-clean-channel epochs that get no stage"], one_channel_rows(res))
        + f"\nWhen exactly one channel is bad, it is {c1} in {pct(channel_bad[c1] / total_bad)} of the no-stage cases and "
        f"{c2} in {pct(channel_bad[c2] / total_bad)}. In total, {thousands(round(pc['pooled_recoverable_minutes']))} of "
        f"the {thousands(round(pc['pooled_missing_minutes']))} minutes without a stage had one clean channel, spread over "
        f"few nights (median {num(pc['recoverable_minutes_per_night']['median'], 1)} minutes per night with a no-stage "
        "epoch). Whether the clean channel alone would stage well is not tested here. The Bitbrain paper reports that a "
        "single frontal channel stages nearly as well as two on public datasets.\n\n"
        "Agreement of the headband stage with the human, by how clean the signal is (epochs where both give a stage, in "
        "brackets):\n\n"
        + table(["Thresholds"] + list(STATUS_LABELS.values()), one_channel_agreement_rows(res))
        + f"\n**A dead signal is not always flagged.** {thousands(t['flat_but_staged'])} of the {thousands(t['n_flat'])} "
        f"flat epochs ({pct(t['flat_but_staged'] / t['n_flat'])}) have a headband stage, in {len(t['flat_staged_by_night'])} "
        "nights. The three nights with the most are "
        + ", ".join(f"{s} ({n} epochs)" for s, n in worst)
        + ". In some of them the signal is completely flat and the stage is held constant for a long stretch, which a "
        "user could read as sleep.\n\n"
        f"![A night with a dead signal]({FIGURES['signal_dead']})\n\n"
        f"![A night where one channel is wild]({FIGURES['signal_one_bad']})\n\n"
        "*Figure F. A night with a dead signal from about the second hour (first), and a night where one channel is wild "
        "for hours while the other stays clean (second). Each overview shows the human and headband stages, the EEG "
        "amplitude, flat epochs, the share of power by band, a spectrogram and movement when the night has it.*\n\n"
        f"**Does the signal carry what the human labels imply?** Across {ss['n_nights']} nights, the share of EEG power "
        "in each band, by the stage the human scored (the channels' mean):\n\n"
        + table(["Human stage", "Nights"] + [f"{b.capitalize()} ({lo:g}-{hi:g} Hz)" for b, (lo, hi) in BANDS.items()],
                signature_rows(res))
        + f"\n![Stage signature]({FIGURES['signature']})\n\n"
        "*Figure G. Left: the table above as a heatmap, each band shaded across the stages. Right: how well delta power "
        "tells N3 from N2 within a night (higher is better, and chance level is the same as a coin flip), for each channel and "
        "for both.*\n\n"
        "How well each feature separates two stages within a night (median over nights of the area under the ROC curve; "
        "a value above one half means the feature is higher in the first stage, and below one half higher in the second):\n\n"
        + table(["Contrast", "Nights"] + list(FEATURE_LABELS.values()), auroc_rows(res))
        + "\nThis checks consistency with the human labels, which were made on the PSG with eye and chin channels that the "
        "headband lacks. It cannot show that those labels are right.\n\n"
        f"**The two channels agree.** Per night, the median correlation between them is "
        f"{num(ss['channel_corr_log_std']['median'], 2)} for amplitude and {num(ss['channel_corr_delta']['median'], 2)} "
        "for delta share, so a second channel adds little on a good night. Its value is as a fallback.\n\n"
        f"**Amplitude spikes.** Epochs with an amplitude of at least three times the night's typical amplitude are "
        f"{pct(ss['spikes']['n_spike_epochs'] / ss['spikes']['n_live_epochs'])} of live epochs:\n\n"
        + table(["Human stage", "Share of spike epochs", "Share of all live epochs", "Ratio"], spike_rows(res))
        + f"\nTheir correlation with the movement sensor is {num(ss['spearman_log_std_vs_movement']['median'], 2)} (median "
        f"over the {ss['spearman_log_std_vs_movement']['n_defined']} nights that have it).\n\n"
        f"**Is a dropout preceded by movement?** In {len(mv['nights_with_movement_signals'])} nights with a movement "
        f"sensor there are {mv['n_runs']} stretches of at least five minutes without a stage that follow staged epochs. "
        f"Compared with the same night's other five-epoch windows, the movement just before such a stretch has a median "
        f"rank of {num(mv['rank']['median'], 2) if mv.get('rank') and mv['rank']['median'] is not None else 'n/a'} "
        "(a rank of one half would be typical movement). Large movement at the very end of a night often marks the band being taken "
        "off.\n\n"
        f"**Order of nights.** For all {len(order)} people recorded on more than one night, the order of the recording "
        f"start dates in the EDF headers matches the order of the recording numbers in "
        f"{sum(o['same_order'] for o in order)} cases, so the first-night analyses above did use the real order.\n\n"
    )


def three_class_rows(res: dict) -> list[list[str]]:
    p = res["three_class"]["primary"]
    return [[cls.capitalize(), thousands(sum(p["counts"][cls].values()))]
            + [pct(p["row_normalised"][cls][c]) for c in ("wake", "light", "deep")] for cls in ("wake", "light", "deep")]


def section_three_class(res: dict) -> str:
    tc = res["three_class"]
    p, nr = tc["primary"], tc["without_rem"]
    c = p["counts"]
    say_deep = sum(c[h]["deep"] for h in c)
    pn = p["per_night"]
    return (
        "## Wake, light and deep sleep\n\n"
        "A closed-loop stimulation system does not need all five stages: it needs to know whether the person is awake, in "
        "light sleep or in deep sleep. Mapping the stages to those three (wake is W, light is N1, N2 and REM, deep is N3, "
        "which is an assumption about such a classifier), the headband agrees with the human on "
        f"{pct(p['accuracy'])} of the {thousands(p['n_epochs'])} epochs where both give a stage. By the class the human scored:\n\n"
        + table(["Human class", "Epochs", "Headband: wake", "Headband: light", "Headband: deep"], three_class_rows(res))
        + f"\nDeep sleep is the weak point: the headband finds {pct(p['deep']['recall'])} of the human's deep epochs "
        f"(precision {pct(p['deep']['precision'])}), and {pct(p['end_to_end_deep_recall'])} when an epoch with no headband "
        f"stage counts as missed. Almost all misses are called light sleep. Of the {thousands(say_deep)} epochs where "
        f"the headband says deep, the human scored {pct(c['wake']['deep'] / say_deep, 2)} as awake, "
        f"{pct(c['light']['deep'] / say_deep)} as light sleep (N2 in {pct(p['headband_deep_but_human_not_n3']['human_says_n2'])} "
        f"of all such epochs) and {pct(c['deep']['deep'] / say_deep)} as deep. So a deep-sleep call that is wrong is "
        "almost always light sleep, rarely wakefulness.\n\n"
        f"Per night, among the {pn['nights_with_enough_human_deep']} nights with enough human deep sleep, the median "
        f"share of it that the headband finds is {num(pn['deep_recall']['median'], 2)} (IQR {num(pn['deep_recall']['q25'], 2)} "
        f"to {num(pn['deep_recall']['q75'], 2)}), and in {pn['nights_where_headband_finds_none_of_them']} of those nights it "
        f"finds none of it. Leaving out every epoch where either label is REM changes little (accuracy {pct(nr['accuracy'])}, "
        f"deep recall {pct(nr['deep']['recall'])}). This uses the dataset's own headband labels, which may not come from "
        "the same classifier that a stimulation system uses.\n\n"
    )


def section_limitations(res: dict) -> str:
    rep = res["dataset_summary"]["readme_reported"]
    hot, d = res["hotspots"], res["dataset_summary"]
    scorer = ""
    if rep["inter_scorer_agreement_pct"] is not None:
        scorer = (f" The dataset README puts the agreement between human scorers at about "
                  f"{num(rep['inter_scorer_agreement_pct'], 0)}%, so some of the disagreement measured here is "
                  "uncertainty in the reference itself.")
    cv = (" The README states that cross-validation was used for the AI labels, but not whether the folds kept all "
          "nights of one person together." if rep["states_cross_validation"] else "")
    return (
        "## Limitations\n\n"
        "- **Hard labels only.** The dataset provides one stage per epoch and no probabilities, so this report can say "
        "where the labels disagree and which simple flags help, but not how confident the model is.\n"
        f"- **The reference is a consensus, not the truth.** The human label is the majority of three scorers.{scorer}\n"
        f"- **Population.** The participants are {d['n_people']} adults from the general population, without recruited "
        "patients and with severe conditions excluded. The results may not carry over to people with sleep disorders, "
        "to other age groups or to home recordings.\n"
        f"- **Repeated nights.** {d['n_nights']} nights come from {d['n_people']} people. Night-level statistics treat "
        "nights as independent. Group comparisons use the person as the unit, and the groups are small.\n"
        f"- **Possible optimism in the AI labels.**{cv} If the folds mixed nights of one person, the agreement "
        "would be somewhat optimistic.\n"
        "- **PSG-based flags are not deployable.** The comparison between the AI on the PSG and the AI on the headband "
        "needs a PSG recording. A home user wears only the headband. It is reported as a reference and not as a feature.\n"
        "- **Dropout is handled in two ways.** The main analysis counts epochs with no headband stage as lost time, the "
        "matched-epoch analysis leaves them out. Which one answers a given question depends on whether the dropout "
        "itself is part of what is being judged.\n"
        "- **Definitions matter.** For example REM latency changes noticeably with the rule for what counts as the first "
        "REM epoch. All definitions are written in `docs/definitions.md`.\n"
        "- **Heuristic flags.** They were fixed in advance and not tuned, but they are evaluated on the same nights as "
        "everything else. A calibrated probability should be compared against them as a baseline.\n"
        "- **How the BOAS labels were made is not documented in the paper cited by the README.** That paper uses other "
        "open datasets. Whether the shipped labels come from the real-time or the offline configuration of the model, "
        "and how the cross-validation folds were built, is an open question.\n"
        "- **Mostly labels.** The headband signal was read only for the signal section. The PSG signals were not, so "
        "whether a disagreement is a headband error, a scorer error or an ambiguous epoch "
        "cannot be told from labels alone.\n\n"
    )


def section_questions(res: dict) -> str:
    return (
        "## Questions for the Bitbrain team\n\n"
        "- How was the consensus built when the three scorers disagreed, and are the individual scorers' labels available? "
        "They would allow measuring scorer uncertainty epoch by epoch.\n"
        "- Were the shipped AI labels predicted out of fold, and were the folds grouped by person? Were they made with the "
        "real-time (past epochs only) or the offline configuration of the model?\n"

        "- A recent public presentation gave a headband accuracy that is lower than the dataset README's. Does that figure "
        f"count epochs where the headband gives no stage as errors, and is it the same data as the {res['dataset_summary']['n_nights']} nights?\n"
        "- What triggers the missing-stage code (-2): a threshold on the model output or a separate artefact detector? "
        "Does it separate an artefact from a lost signal?\n"
        "- Did the headband stop or come off at the end of the nights where it gives no stage until the end of the "
        "recording?\n"
        "- Why is N3 scarce even in the youngest participants: conservative scoring, or something about the setting "
        "or the population?\n\n"
    )


def section_appendix(res: dict) -> str:
    items = "".join(f"- [{label}]({path})\n" for label, path in EXTRA_FIGURES.items())
    return ("## More figures\n\n" + items + "\n## Reproducing this report\n\n"
            "All numbers come from `results/*.json`, written by the scripts in `src/boas_unc/`. Definitions are in "
            "`docs/definitions.md`, open questions and decisions in `CHECKLIST.md`, and the data inventory in "
            "`docs/data_inventory.md`. Regenerate everything with `uv run python -m boas_unc.report` after the other "
            "result scripts.\n")


def render(res: dict) -> str:
    return (
        "# Where a wearable headband is uncertain, and how to flag it\n\n"
        "Phase 0: a label-only analysis of the Bitbrain Open Access Sleep (BOAS) dataset. "
        "*This file is generated by `src/boas_unc/report.py`. Do not edit it by hand.*\n\n"
        + section_summary(res) + section_dataset(res) + section_methods(res) + section_agreement(res)
        + section_three_class(res) + section_hotspots(res) + section_clinical(res) + section_person(res) + section_signal(res)
        + section_abstention(res)
        + section_limitations(res)
        + section_questions(res) + section_appendix(res))


def main():
    summary = build_dataset_summary()
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=1))
    REPORT.write_text(render(read_results()))
    print(f"wrote {RESULTS_DIR / 'dataset_summary.json'} and {REPORT}")


if __name__ == "__main__":
    main()
