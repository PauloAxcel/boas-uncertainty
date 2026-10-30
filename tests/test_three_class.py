import numpy as np
import pandas as pd
import pytest

from boas_unc import three_class as tc

A = np.array


def test_to_three_maps_stages_and_marks_the_rest():
    assert tc.to_three(A([0, 1, 2, 3, 4, 8, -2])).tolist() == [0, 1, 1, 2, 1, -1, -1]


def test_confusion_and_precision_recall():
    human, hb = A([2, 2, 2, 2, 1, 0]), A([2, 2, 1, 0, 1, 0])
    m = tc.confusion(human, hb)
    assert m.sum() == 6 and m[2, 2] == 2 and m[2, 1] == 1 and m[2, 0] == 1
    p, r = tc.precision_recall(m, 2)
    assert p == 1.0 and r == 0.5
    assert np.isnan(tc.precision_recall(np.zeros((3, 3), dtype=int), 2)[0])


def test_f1_guards():
    assert np.isnan(tc.f1(np.nan, 0.5)) and tc.f1(0.0, 0.0) == 0.0 and tc.f1(0.5, 1.0) == pytest.approx(2 / 3)


def test_summarise_confusion_pooled_numbers():
    m = np.array([[8, 2, 0], [1, 18, 1], [0, 5, 5]])
    out = tc.summarise_confusion(m)
    assert out["n_epochs"] == 40 and out["accuracy"] == pytest.approx(31 / 40)
    assert out["deep"]["recall"] == 0.5 and out["deep"]["precision"] == pytest.approx(5 / 6)
    assert out["wake"]["precision"] == pytest.approx(8 / 9)
    assert out["row_normalised"]["deep"]["light"] == 0.5


def table(hum, hb, sub="a"):
    n = len(hum)
    return pd.DataFrame({"sub": sub, "psg_hum": hum, "hb_ai": hb, "epoch": range(n)})


def test_evaluate_counts_a_deep_epoch_the_headband_calls_n2_as_missed_and_reports_what_the_human_said():
    hum = [3] * 25 + [2] * 25
    hb = [3] * 10 + [2] * 15 + [3] * 5 + [2] * 20  # 10 of 25 deep found; 5 false deep calls on human N2
    out = tc.evaluate(table(hum, hb))
    assert out["deep"]["recall"] == pytest.approx(10 / 25) and out["deep"]["precision"] == pytest.approx(10 / 15)
    assert out["headband_deep_but_human_not_n3"]["share"] == pytest.approx(5 / 15)
    assert out["headband_deep_but_human_not_n3"]["human_says_n2"] == pytest.approx(5 / 15)
    assert out["per_night"]["nights_with_enough_human_deep"] == 1


def test_end_to_end_recall_counts_a_missing_headband_stage_as_missed():
    hum = [3] * 30
    hb = [3] * 15 + [-2] * 15
    out = tc.evaluate(table(hum, hb))
    assert out["deep"]["recall"] == 1.0  # on the epochs where the headband gives a stage
    assert out["end_to_end_deep_recall"] == pytest.approx(0.5)


def test_night_with_enough_deep_but_none_found_is_counted():
    hum = [3] * 30 + [2] * 30
    hb = [2] * 60
    assert tc.evaluate(table(hum, hb))["per_night"]["nights_where_headband_finds_none_of_them"] == 1


def test_leaving_out_rem_drops_the_epochs_where_either_label_is_rem():
    hum = [4] * 10 + [2] * 10 + [3] * 10
    hb = [2] * 10 + [4] * 10 + [3] * 10
    assert tc.evaluate(table(hum, hb))["n_epochs"] == 30
    assert tc.evaluate(table(hum, hb), leave_out_rem=True)["n_epochs"] == 10


def test_run_has_both_versions():
    out = tc.run(table([3] * 25 + [2] * 25, [3] * 25 + [2] * 25))
    assert set(out) >= {"primary", "without_rem", "mapping"} and out["primary"]["accuracy"] == 1.0
