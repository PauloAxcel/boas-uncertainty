import numpy as np
import pandas as pd
import pytest

from boas_unc import hotspots as hs

A = np.array
INF = np.inf


def test_distance_to_next_and_from_prev():
    mask = A([False, False, True, False, False, True, False])
    assert hs.distance_to_next(mask).tolist() == [2, 1, 0, 2, 1, 0, INF]
    assert hs.distance_from_prev(mask).tolist() == [INF, INF, 0, 1, 2, 0, 1]


def test_distances_without_any_true_are_infinite():
    none = np.zeros(3, dtype=bool)
    assert np.isinf(hs.distance_to_next(none)).all() and np.isinf(hs.distance_from_prev(none)).all()


def test_distance_to_nearest_takes_the_closer_side():
    mask = A([False, False, True, False, False, False, False, True])
    assert hs.distance_to_nearest(mask).tolist() == [2, 1, 0, 1, 2, 2, 1, 0]


def test_transition_mask_marks_both_sides_of_a_change():
    assert hs.transition_mask(A([2, 2, 2, 3, 3, 3])).tolist() == [False, False, True, True, False, False]


def test_transition_mask_ignores_boundaries_with_code_8():
    assert not hs.transition_mask(A([2, 2, 8, 8, 3, 3])).any()  # into, out of and across a gap
    assert hs.transition_mask(A([2, 3, 8, 3])).tolist() == [True, True, False, False]


def test_cap_distance_groups_far_and_infinite():
    assert hs.cap_distance(A([0, 1, 5, 6, 40, INF]), max_k=5).tolist() == [0, 1, 5, 6, 6, 6]


def test_hour_of_night_floors_and_caps():
    assert hs.hour_of_night(A([0, 3599, 3600, 7200, 9 * 3600, 9.5 * 3600])).tolist() == [0, 0, 1, 2, 9, 9]


def night(hum, hb, sub="s1"):
    n = len(hum)
    return pd.DataFrame({"sub": sub, "onset": np.arange(n) * 30, "psg_hum": hum, "hb_ai": hb,
                         "hb_ai_missing": np.array(hb) == -2})


def test_annotate_night_adds_distances_and_hour():
    out = hs.annotate_night(night([2, 2, 2, 3, 3], [2, 2, -2, 3, 3]))
    assert out.dist_transition.tolist() == [2, 1, 0, 0, 1]
    assert out.to_next_flag.tolist() == [2, 1, 0, INF, INF]
    assert out.since_prev_flag.tolist() == [INF, INF, 0, 1, 2]
    assert out.hour.eq(0).all()


def test_annotate_stacks_nights_independently():
    t = pd.concat([night([2, 3], [2, 3], "a"), night([1, 1, 1], [1, 1, 1], "b")], ignore_index=True)
    out = hs.annotate(t)
    assert out[out["sub"] == "b"].dist_transition.tolist() == [6, 6, 6]


def test_both_stage_excludes_human_8_and_headband_minus2():
    df = night([0, 8, 2, 2], [0, 0, -2, 2])
    assert hs.both_stage(df).tolist() == [True, False, False, True]


def test_rate_summary_counts_missing_separately_from_disagreement():
    df = night([0, 8, 2, 2, 3], [0, 0, -2, 1, 3])
    out = hs.rate_summary(df)
    assert out["n_epochs"] == 3 and out["n_disagree"] == 1
    assert out["disagreement_rate"] == pytest.approx(1 / 3)
    assert out["n_human_epochs"] == 4 and out["headband_missing_rate"] == pytest.approx(1 / 4)


def test_rate_summary_empty_group_is_nan():
    out = hs.rate_summary(night([8], [0]))
    assert np.isnan(out["disagreement_rate"]) and out["n_epochs"] == 0


def test_night_rates_needs_enough_epochs():
    t = pd.concat([night([2] * 10, [2] * 9 + [3], "a"), night([2] * 9, [3] * 9, "b")], ignore_index=True)
    rates = hs.night_rates(t, min_epochs=10)
    assert rates.index.tolist() == ["a", "b"] and rates["a"] == pytest.approx(0.1)
    assert np.isnan(rates["b"])  # too few epochs: kept as undefined, not dropped


def test_night_rates_skips_nights_with_no_version_a_epoch():
    t = pd.concat([night([2] * 10, [2] * 10, "a"), night([2, 2], [-2, -2], "b")], ignore_index=True)
    assert hs.night_rates(t, min_epochs=1).index.tolist() == ["a"]


def test_group_summary_and_by_key():
    t = pd.concat([night([2] * 10, [2] * 10, "a"), night([3] * 10, [2] * 10, "b")], ignore_index=True)
    t["g"] = t["sub"].map({"a": "x", "b": "y"})
    out = hs.by_key(t, "g")
    assert out["x"]["disagreement_rate"] == 0.0 and out["y"]["disagreement_rate"] == 1.0
    assert out["y"]["per_night_rate"]["n_defined"] == 1


def test_group_summary_counts_nights_with_too_few_epochs_as_undefined():
    t = pd.concat([night([2] * 10, [2] * 10, "a"), night([2] * 3, [3] * 3, "b")], ignore_index=True)
    out = hs.group_summary(t)["per_night_rate"]
    assert out["n_defined"] == 1 and out["n_undefined"] == 1


def test_defined_plus_undefined_equals_n_nights():
    t = pd.concat([night([2] * 10, [2] * 10, "a"), night([2] * 3, [3] * 3, "b"),
                   night([2, 2], [-2, -2], "c")], ignore_index=True)
    out = hs.group_summary(t)
    assert out["n_nights"] == 2
    assert out["per_night_rate"]["n_defined"] + out["per_night_rate"]["n_undefined"] == out["n_nights"]


def test_labels():
    assert hs.distance_label(3) == "3" and hs.distance_label(6) == "6+"
    assert hs.hour_label(8) == "8" and hs.hour_label(9) == "9+"


def test_by_true_stage_orders_stages_and_has_confusion():
    out = hs.by_true_stage(night([0, 0, 2, 2, 4], [0, 2, 2, 2, 4]))
    assert list(out["by_stage"]) == ["W", "N2", "REM"]
    assert out["by_stage"]["W"]["disagreement_rate"] == 0.5
    assert out["confusion_counts_version_a"]["W"]["N2"] == 1


def test_by_distance_and_stage_by_distance_cover_the_bins():
    df = hs.annotate_night(night([2, 2, 2, 3, 3, 3, 3], [2, 2, 2, 3, 3, 3, 3]))
    assert set(hs.by_distance_to_transition(df)) == {"0", "1", "2", "3"}  # distances 2,1,0,0,1,2,3
    grid = hs.stage_by_distance(df)
    assert set(grid) == {"W", "N1", "N2", "N3", "REM"} and set(grid["N2"]) == {"0", "1", "2", "3", "4", "5", "6+"}
    assert grid["N2"]["0"]["n_epochs"] == 1


def test_by_artefact_distance_excludes_flagged_epochs_from_the_rate():
    # epoch 2 is -2; epoch 1 is before it, epoch 3 after it; epochs 0 and 4 are within 5 epochs too
    df = hs.annotate_night(night([2, 2, 2, 2, 2], [2, 3, -2, 2, 2]))
    out = hs.by_artefact_distance(df)
    assert out["before_flag"]["1"]["disagreement_rate"] == 1.0  # epoch 1 disagrees
    assert out["after_flag"]["1"]["disagreement_rate"] == 0.0
    assert out["no_flag_within_5"]["n_epochs"] == 0


def test_per_person_averages_a_persons_nights():
    rates = pd.Series({"s1": 0.1, "s2": 0.3, "s3": 0.2})
    pid = pd.Series({"s1": "p", "s2": "p", "s3": "q"})
    out = hs.per_person(rates, pid)
    assert out["p"] == pytest.approx(0.2) and out["q"] == pytest.approx(0.2)


def test_person_groups_reports_nights_and_people_and_skips_missing_group():
    rates = pd.Series({"s1": 0.1, "s2": 0.3, "s3": 0.2, "s4": 0.9})
    pid = pd.Series({"s1": "p", "s2": "p", "s3": "q", "s4": "r"})
    group = pd.Series({"s1": "F", "s2": "F", "s3": "M", "s4": None})
    out = hs.person_groups(rates, pid, group, ["M", "F"])
    assert list(out) == ["M", "F"]  # the order asked for, not first appearance
    assert out["F"]["n_nights"] == 2 and out["F"]["n_pids"] == 1
    assert out["M"]["n_pids"] == 1 and "None" not in out and len(out) == 2


def test_by_distance_to_transition_keys_are_in_natural_order():
    df = hs.annotate_night(night([2, 2, 2, 3, 3, 3, 3], [2] * 7))
    assert list(hs.by_distance_to_transition(df)) == ["0", "1", "2", "3"]


def test_night_missing_rates_uses_human_stage_epochs():
    t = pd.concat([night([2, 2, 8, 2], [2, -2, -2, 2], "a"), night([1, 1], [1, 1], "b")], ignore_index=True)
    out = hs.night_missing_rates(t)
    assert out["a"] == pytest.approx(1 / 3) and out["b"] == 0.0


def test_age_band():
    assert hs.age_band("18") == "18-29" and hs.age_band("44") == "30-44" and hs.age_band("71") == "60+"
    assert hs.age_band("n/a") is None and hs.age_band("5") is None


def test_participant_groups_handles_na_age():
    p = pd.DataFrame({"participant_id": ["sub-1", "sub-2"], "age": ["27", "n/a"],
                      "sex": ["F", "n/a"], "pid": ["1", "2"]})
    pid, band, sex = hs.participant_groups(p)
    assert band.tolist()[0] == "18-29" and pd.isna(band.tolist()[1])
    assert sex.tolist()[0] == "F" and pd.isna(sex.tolist()[1])


def test_run_has_all_sections():
    t = pd.concat([night([2] * 12 + [3] * 12, [2] * 11 + [3] * 13, "sub-1"),
                   night([0] * 12 + [4] * 12, [0] * 12 + [4] * 12, "sub-2")], ignore_index=True)
    p = pd.DataFrame({"participant_id": ["sub-1", "sub-2"], "age": ["27", "n/a"],
                      "sex": ["F", "M"], "pid": ["1", "2"]})
    out = hs.run(t, p)
    assert {"by_stage", "by_headband_artefact_flag", "by_hour_of_night", "by_age_band", "by_sex",
            "per_participant"} <= set(out)
    assert out["n_nights_age_unknown"] == 1
    assert out["by_age_band"]["18-29"]["n_pids"] == 1
