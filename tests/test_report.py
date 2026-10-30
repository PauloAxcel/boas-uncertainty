import ast
import copy
import re
from pathlib import Path

import pandas as pd
import pytest

from boas_unc import report as rp

SOURCE = Path(rp.__file__)


def prose_constants(tree: ast.AST):
    """String constants that are prose: not docstrings and not format specs such as '.1f'."""
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            skip |= {id(n) for n in ast.walk(node.format_spec)}
        if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip]


# Names, not results: stage names, the F1 metric, the dataset codes, and the phase label.
ALLOWED_LABELS = re.compile(r"\b(?:N[123]|F1)\b|HB_[12]|\(-2\)|a -2|-2 |code -?\d|Phase 0|Figure [A-Z]")


def typed_numbers(tree: ast.AST) -> list[str]:
    """Prose (a string with a space) containing a digit that is not an allowed label."""
    return [n.value for n in prose_constants(tree)
            if re.search(r"\s", n.value.strip()) and re.search(r"\d", ALLOWED_LABELS.sub("", n.value))]


def test_no_number_is_typed_into_the_report_text():
    """File names, keys and labels have no space, so only prose is checked."""
    assert typed_numbers(ast.parse(SOURCE.read_text())) == []


def test_the_scan_catches_a_typed_number():
    tree = ast.parse("x = 'agreement is 87 percent'\ny = f'{v:.1f} of epochs'\nz = 'fig5_name.png'")
    assert typed_numbers(tree) == ["agreement is 87 percent"]
    assert typed_numbers(ast.parse("x = 'N1 and N3 are hard, F1 is low, code 8 and -2 are flags'")) == []


def test_formatters():
    assert rp.pct(0.8674) == "86.7%" and rp.pct(0.86741, 2) == "86.74%" and rp.pct(None) == "n/a"
    assert rp.num(3.14159, 2) == "3.14" and rp.num(None) == "n/a"
    assert rp.signed(1.26) == "+1.3" and rp.signed(-0.04) == "-0.0" and rp.signed(None) == "n/a"
    assert rp.thousands(120095) == "120,095"
    s = {"median": 0.5, "q25": 0.25, "q75": 0.75}
    assert rp.med_iqr(s, 100, 0, "%") == "50% (25 to 75)"
    assert rp.med_iqr({"median": None, "q25": None, "q75": None}) == "n/a"
    assert rp.join_names(["a", "b", "c"]) == "a, b, and c" and rp.join_names(["a"]) == "a"
    assert rp.epochs(1) == "1 epoch" and rp.epochs("3") == "3 epochs"


def test_table_has_header_separator_and_rows():
    out = rp.table(["A", "B"], [["1", "2"], ["3", "4"]])
    assert out.splitlines() == ["| A | B |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_flag_label_covers_every_kind():
    assert rp.flag_label("none").startswith("No abstention")
    assert rp.flag_label("change_0") == "On a predicted stage change"
    assert rp.flag_label("change_1") == "Within 1 epoch of a predicted stage change"
    assert rp.flag_label("artefact_3").startswith("Within 3 epochs of a headband")
    assert rp.flag_label("combined_2").startswith("Within 2 epochs")
    assert "NOT available on the device" in rp.flag_label("psg_ai_disagrees")


def test_readme_facts_reads_the_figures_from_the_text():
    text = ("estimated inter-scorer agreement of approximately 82% (x). The model achieved an 87.13% match between "
            "human-consensus and network-provided labels for the PSG data, and an 86.71% match for the wearable EEG "
            "data. We used cross-validation.\nEsparza-Iaizzo, M. (2024). Title. doi: 10.1016/j.x.2026.111560.\n")
    facts = rp.readme_facts(text)
    assert facts["inter_scorer_agreement_pct"] == 82.0
    assert facts["human_vs_psg_ai_match_pct"] == 87.13 and facts["human_vs_headband_ai_match_pct"] == 86.71
    assert facts["model_paper_doi"] == "10.1016/j.x.2026.111560" and facts["states_cross_validation"]
    empty = rp.readme_facts("nothing here")
    assert empty["inter_scorer_agreement_pct"] is None and empty["model_paper_doi"] is None


def test_build_dataset_summary_from_a_tiny_dataset(tmp_path):
    (tmp_path / "dataset_description.json").write_text(
        '{"Name": "Tiny", "DatasetDOI": "doi:x", "License": "CC0", "Authors": ["A", "B"]}')
    (tmp_path / "README").write_text("inter-scorer agreement of approximately 80%")
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\tbmi\tpid\nsub-1\t30\tF\t20\t1\nsub-2\tn/a\tM\t21\t2\n"
        "sub-3\t30\tF\t22\t1\nsub-4\t50\tM\t23\t3\n")
    for sub in ("sub-1", "sub-2", "sub-3", "sub-4"):
        eeg = tmp_path / sub / "eeg"
        eeg.mkdir(parents=True)
        psg = pd.DataFrame({"onset": [0, 30, 60], "stage_hum": [0, 2, 2], "stage_ai": [0, 2, 2]})
        hb = pd.DataFrame({"onset": [0, 30, 60], "stage_ai": [0, 2, -2]})
        psg.to_csv(eeg / f"{sub}_task-Sleep_acq-psg_events.tsv", sep="\t", index=False)
        hb.to_csv(eeg / f"{sub}_task-Sleep_acq-headband_events.tsv", sep="\t", index=False)
    out = rp.build_dataset_summary(tmp_path)
    assert out["n_nights"] == 4 and out["n_people"] == 3
    assert out["n_nights_age_unknown"] == 1 and out["n_people_age_unknown"] == 1
    assert out["age_nights_min_median_max"] == [30.0, 30.0, 50.0]  # counted by night
    assert out["age_people_min_median_max"] == [30.0, 40.0, 50.0]  # counted by person: the median moves
    assert out["epoch_seconds"] == 30.0
    assert out["sex_nights"] == {"F": 2, "M": 2} and out["sex_people"] == {"F": 1, "M": 2}
    assert out["readme_reported"]["inter_scorer_agreement_pct"] == 80.0


@pytest.fixture(scope="module")
def results():
    try:
        return rp.read_results()
    except FileNotFoundError:
        pytest.skip("run the result scripts first")


def test_every_figure_exists():
    paths = list(rp.FIGURES.values()) + list(rp.EXTRA_FIGURES.values())
    assert [p for p in paths if not (rp.ROOT / p).exists()] == []


def test_report_renders_without_placeholders(results):
    text = rp.render(results)
    assert "None" not in text and not re.search(r"\bnan\b", text, re.I) and "{" not in text
    for heading in ("## Summary", "## Dataset and data used", "## Methods", "## Agreement", "## Limitations"):
        assert heading in text
    assert all(rp.FIGURES[k] in text for k in rp.FIGURES)


def test_numbers_in_the_report_follow_the_results_files(results):
    changed = copy.deepcopy(results)
    changed["dataset_summary"]["n_nights"] = 7771
    changed["dataset_summary"]["n_people"] = 8882
    changed["abstention"]["baseline_accuracy"] = 0.4321
    changed["clinical"]["standalone"]["tst"]["bias"] = -123.4
    text = rp.render(changed)
    assert "7771 nights from 8882 people" in text and "43.2%" in text and "-123.4 min" in text
    assert "7771" not in rp.render(results)


def test_summary_names_the_hardest_stage_from_the_data(results):
    changed = copy.deepcopy(results)
    for g in changed["hotspots"]["by_stage"].values():
        g["disagreement_rate"] = 0.1
    changed["hotspots"]["by_stage"]["REM"]["disagreement_rate"] = 0.9
    assert "REM is the hardest stage: 90.0%" in rp.section_summary(changed)


def test_multiple_comparison_note_is_computed_from_the_results(results):
    changed = copy.deepcopy(results)
    changed["person_effects"]["c_sex_bmi_age"]["disagreement_rate"]["age"]["p"] = 0.0001
    assert "(disagreement rate against age)" in rp.multiple_comparison_note(changed)
    assert "is below it" in rp.multiple_comparison_note(changed)
    changed["person_effects"]["c_sex_bmi_age"]["disagreement_rate"]["age"]["p"] = 0.9
    assert "is above it" in rp.multiple_comparison_note(changed)


def test_smoothing_sentence_says_stays_or_moves():
    same = {"agreement_version_a": {"before": {"accuracy": 0.86741, "kappa": 0.7692}, "after": {"accuracy": 0.86741, "kappa": 0.7687}}}
    moved = {"agreement_version_a": {"before": {"accuracy": 0.86, "kappa": 0.7}, "after": {"accuracy": 0.88, "kappa": 0.75}}}
    assert rp.smoothing_agreement(same).startswith("Agreement stays at")
    assert rp.smoothing_agreement(moved).startswith("Agreement moves from")


def test_report_file_matches_the_generator(results):
    if rp.REPORT.exists():
        assert rp.REPORT.read_text() == rp.render(results)
