import json

import numpy as np
import pandas as pd
import pytest

from boas_unc import agreement as ag

A = np.array


def test_is_stage_excludes_special_codes_and_nan():
    assert ag.is_stage(A([0, 1, 4, 8, -2, np.nan])).tolist() == [True, True, True, False, False, False]


def test_version_mask_a_needs_both_b_needs_reference():
    ref, cand = A([0, 1, 2, -2, 3]), A([0, -2, 2, 3, -2])
    assert ag.version_mask(ref, cand, "a_both_stage").tolist() == [True, False, True, False, False]
    assert ag.version_mask(ref, cand, "b_end_to_end").tolist() == [True, True, True, False, True]


def test_version_mask_rejects_unknown_version():
    with pytest.raises(ValueError):
        ag.version_mask(A([0]), A([0]), "c")


def test_accuracy_counts_equal_labels_and_handles_empty():
    assert ag.accuracy(A([0, 1, 2, 3]), A([0, 1, 2, 4])) == 0.75
    assert np.isnan(ag.accuracy(A([]), A([])))


def test_kappa_perfect_and_chance_and_single_class():
    assert ag.kappa(A([0, 1, 2, 3]), A([0, 1, 2, 3])) == 1.0
    assert ag.kappa(A([0, 0, 1, 1]), A([0, 1, 0, 1])) == 0.0
    assert np.isnan(ag.kappa(A([2, 2, 2]), A([2, 2, 2])))


def test_kappa_end_to_end_counts_missing_as_disagreement_with_no_chance_agreement():
    ref, cand = A([0, 0, 1, 1]), A([0, 0, 1, -2])
    assert ag.accuracy(ref, cand) == 0.75
    assert ag.kappa(ref, cand) < 0.75  # chance-corrected, and -2 can never match by chance


def test_coverage_is_share_of_reference_stages_with_a_candidate_stage():
    ref, cand = A([0, 1, 2, 3, -2]), A([0, -2, 2, -2, 4])
    assert ag.coverage(ref, cand) == 0.5
    assert np.isnan(ag.coverage(A([-2]), A([1])))


def test_per_stage_scores_missing_candidate_lowers_recall_not_precision():
    ref, cand = A([2, 2, 2, 2]), A([2, 2, -2, -2])
    n2 = ag.per_stage_scores(ref, cand)["N2"]
    assert n2["precision"] == 1.0 and n2["recall"] == 0.5 and n2["support"] == 4


def test_per_stage_scores_undefined_stage_is_nan_not_zero():
    ref, cand = A([0, 0, 2]), A([0, 0, 2])
    n3 = ag.per_stage_scores(ref, cand)["N3"]  # absent from both sequences
    assert np.isnan(n3["precision"]) and np.isnan(n3["recall"]) and np.isnan(n3["f1"])
    # present in the reference but never predicted: recall 0, precision undefined, F1 0 (not dropped)
    n2 = ag.per_stage_scores(A([2, 2]), A([0, 0]))["N2"]
    assert n2["recall"] == 0.0 and np.isnan(n2["precision"]) and n2["f1"] == 0.0
    # predicted but absent from the reference: precision 0, recall undefined, F1 0
    n3 = ag.per_stage_scores(A([0, 0]), A([3, 0]))["N3"]
    assert n3["precision"] == 0.0 and np.isnan(n3["recall"]) and n3["f1"] == 0.0


def test_per_stage_f1_is_the_count_form():
    ref, cand = A([2, 2, 2, 2, 0]), A([2, 2, 0, 0, 2])  # N2: TP 2, FP 1, FN 2
    assert ag.per_stage_scores(ref, cand)["N2"]["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 2))


def test_confusion_counts_shape_and_missing_column():
    counts = ag.confusion_counts(A([0, 0, 1, 1]), A([0, 1, 1, -2]))
    assert list(counts.index) == ["W", "N1", "N2", "N3", "REM"]
    assert list(counts.columns) == ["W", "N1", "N2", "N3", "REM", "-2"]
    assert counts.loc["W", "W"] == 1 and counts.loc["W", "N1"] == 1
    assert counts.loc["N1", "N1"] == 1 and counts.loc["N1", "-2"] == 1
    assert counts.to_numpy().sum() == 4


def test_row_normalise_rows_sum_to_one_and_empty_rows_are_nan():
    norm = ag.row_normalise(ag.confusion_counts(A([0, 0, 1, 1]), A([0, 1, 1, -2])))
    assert norm.loc["W"].sum() == pytest.approx(1.0) and norm.loc["N1"].sum() == pytest.approx(1.0)
    assert norm.loc["N3"].isna().all()


def test_evaluate_epochs_collects_the_pooled_metrics():
    out = ag.evaluate_epochs(A([0, 1, 2, 2]), A([0, 1, 2, 3]))
    assert out["n_epochs"] == 4 and out["n_agree"] == 3 and out["accuracy"] == 0.75
    assert set(out) >= {"kappa", "per_stage", "confusion_counts", "confusion_row_normalised"}


def test_summarise_ignores_nan_and_reports_how_many():
    s = ag.summarise(pd.Series([1.0, 2.0, 3.0, np.nan]))
    assert s["n_defined"] == 3 and s["n_undefined"] == 1 and s["median"] == 2.0
    assert np.isnan(ag.summarise(pd.Series([np.nan]))["median"])


def test_night_scores_has_flat_keys():
    out = ag.night_scores(A([0, 2, 2]), A([0, 2, 3]))
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert {"kappa", "W_recall", "N2_precision", "REM_f1"} <= set(out)


def make_table():
    """Two nights. Night b has a human-8 epoch, a headband -2 and a PSG AI -2."""
    rows = [("a", 0, 0, 0, False), ("a", 1, 1, 1, False), ("a", 2, 2, 3, False),
            ("b", 0, 0, 0, False), ("b", 8, 0, 0, True), ("b", 2, -2, -2, False), ("b", 3, 3, -2, False)]
    return pd.DataFrame(rows, columns=["sub", "psg_hum", "psg_ai", "hb_ai", "hum_disconnected"])


def test_per_night_one_row_per_night_for_each_version():
    scores = ag.per_night(make_table(), "psg_hum", "hb_ai", "b_end_to_end")
    assert scores.index.tolist() == ["a", "b"]
    assert scores.loc["a", "accuracy"] == pytest.approx(2 / 3)


def test_evaluate_pair_excludes_human_8_and_counts_candidate_missing_in_b():
    out = ag.evaluate_pair(make_table(), "psg_hum", "hb_ai")
    assert out["n_epochs_after_excluding_human_8"] == 6
    # (a) night a: 3 epochs, 2 agree; night b: 1 epoch with both stages, 1 agrees; -2 epochs drop out
    assert out["a_both_stage"]["pooled"]["n_epochs"] == 4
    assert out["a_both_stage"]["pooled"]["n_agree"] == 3
    assert out["b_end_to_end"]["pooled"]["n_epochs"] == 6
    assert out["coverage"]["pooled"] == pytest.approx(4 / 6)


def test_evaluate_pair_with_psg_ai_reference_drops_its_missing_epochs():
    out = ag.evaluate_pair(make_table(), "psg_ai", "hb_ai")
    # PSG AI -2 (night b, epoch 2) is no reference: dropped from (b), not an error
    assert out["b_end_to_end"]["pooled"]["n_epochs"] == 5


def test_jsonable_turns_nan_into_none_and_numpy_into_python():
    out = ag.jsonable({"a": np.nan, "b": np.int64(3), "c": [np.float64(0.12345678)], 1: (1, 2)})
    assert out == {"a": None, "b": 3, "c": [0.123457], "1": [1, 2]}
    json.dumps(out, allow_nan=False)


def test_run_has_three_pairs():
    out = ag.run(make_table())
    assert out["n_nights"] == 2 and set(out["pairs"]) == set(ag.PAIRS)
