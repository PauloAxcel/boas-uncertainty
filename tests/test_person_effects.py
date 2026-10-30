import numpy as np
import pandas as pd
import pytest

from boas_unc import person_effects as pe


def make_nights(rows):
    """rows: (sub, pid, dropout, rate, sex, age, bmi, se)"""
    df = pd.DataFrame(rows, columns=["sub", "pid", "dropout_share", "disagreement_rate", "sex", "age", "bmi",
                                     "hum_sleep_efficiency"]).set_index("sub")
    df["hum_sol"], df["hum_waso"], df["hum_pct_rem"] = 10.0, 5.0, 20.0
    df["order"] = [int(s.split("-")[1]) for s in df.index]
    return df


def test_icc1_is_one_when_nights_of_a_person_are_identical():
    assert pe.icc1(np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0]), np.array(list("aabbcc"))) == pytest.approx(1.0)


def test_icc1_is_negative_when_people_do_not_differ_but_nights_do():
    assert pe.icc1(np.array([1.0, 3.0] * 3), np.array(list("aabbcc"))) == pytest.approx(-1.0)


def test_icc1_needs_two_groups_with_two_or_more_values():
    assert np.isnan(pe.icc1(np.array([1.0, 2.0, 3.0]), np.array(["a", "a", "a"])))
    assert np.isnan(pe.icc1(np.array([1.0, 2.0, 3.0]), np.array(["a", "b", "c"])))


def test_bootstrap_icc_is_reproducible_and_ordered():
    v, g = np.array([1.0, 1.2, 5.0, 5.1, 9.0, 8.8, 3.0, 3.3]), np.array(list("aabbccdd"))
    first, second = pe.bootstrap_icc(v, g, n_boot=200), pe.bootstrap_icc(v, g, n_boot=200)
    assert first == second and first[0] <= first[1]


def test_repeated_keeps_only_people_with_two_or_more_nights():
    nights = make_nights([("sub-1", "p", 0.0, 0.1, "F", 30, 20, 80), ("sub-2", "q", 0.0, 0.1, "M", 30, 20, 80),
                          ("sub-3", "p", 0.0, 0.1, "F", 30, 20, 80)])
    assert pe.repeated(nights).index.tolist() == ["sub-1", "sub-3"]


def test_pair_values_gives_every_pair_in_order():
    nights = make_nights([(f"sub-{i}", "p", float(i), 0.1, "F", 30, 20, 80) for i in (1, 2, 3)])
    x, y = pe.pair_values(nights, "dropout_share")
    assert list(zip(x, y)) == [(1.0, 2.0), (1.0, 3.0), (2.0, 3.0)]


def test_spearman_guards():
    out = pe.spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0]))
    assert out["rho"] == pytest.approx(1.0) and out["n"] == 4
    assert np.isnan(pe.spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["rho"])
    assert np.isnan(pe.spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))["rho"])
    assert pe.spearman(np.array([1.0, 2.0, np.nan, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))["n"] == 3


def test_high_dropout_nights_lists_the_persons_other_nights():
    nights = make_nights([("sub-1", "p", 0.5, 0.1, "F", 30, 20, 80), ("sub-2", "p", 0.02, 0.1, "F", 30, 20, 80),
                          ("sub-3", "q", 0.3, 0.1, "M", 30, 20, 80), ("sub-4", "r", 0.0, 0.1, "M", 30, 20, 80)])
    out = pe.high_dropout_nights(nights)
    assert [h["night"] for h in out] == ["sub-1", "sub-3"]
    assert out[0]["other_nights"] == {"sub-2": 0.02} and out[1]["other_nights"] == {}


def test_first_vs_later_uses_the_lowest_number_as_first_and_averages_the_rest():
    nights = make_nights([("sub-1", "p", 0.1, 0.1, "F", 30, 20, 70), ("sub-2", "p", 0.0, 0.1, "F", 30, 20, 80),
                          ("sub-3", "p", 0.0, 0.1, "F", 30, 20, 90),
                          ("sub-4", "q", 0.0, 0.1, "M", 30, 20, 85), ("sub-5", "q", 0.0, 0.1, "M", 30, 20, 75)])
    out = pe.first_vs_later(nights, "hum_sleep_efficiency")
    assert out["n_people"] == 2 and out["n_later_higher"] == 1 and out["n_later_lower"] == 1
    assert out["difference"]["median"] == pytest.approx((15.0 + (-10.0)) / 2)  # p: 85 - 70, q: 75 - 85
    assert np.isnan(out["wilcoxon_p"])  # too few people for the test


def test_first_vs_later_skips_people_with_an_undefined_value():
    nights = make_nights([("sub-1", "p", 0.0, 0.1, "F", 30, 20, np.nan), ("sub-2", "p", 0.0, 0.1, "F", 30, 20, 80)])
    assert pe.first_vs_later(nights, "hum_sleep_efficiency")["n_people"] == 0


def test_person_means_and_sex_comparison():
    nights = make_nights([("sub-1", "p", 0.2, 0.1, "F", 30, 20, 80), ("sub-2", "p", 0.4, 0.3, "F", 30, 20, 80),
                          ("sub-3", "q", 0.0, 0.1, "M", 40, 25, 80), ("sub-4", "r", 0.1, 0.1, "M", 50, 24, 80),
                          ("sub-5", "s", 0.6, 0.1, "F", 60, 22, 80)])
    people = pe.person_means(nights, ["dropout_share", "age", "bmi"])
    assert people.loc["p", "dropout_share"] == pytest.approx(0.3) and people.loc["p", "n_nights"] == 2
    out = pe.sex_comparison(people, "dropout_share")
    assert out["groups"]["F"]["n_people"] == 2 and out["groups"]["M"]["n_people"] == 2
    assert 0 <= out["mannwhitney_p"] <= 1


def test_covariate_correlation_counts_people_without_a_value():
    people = pd.DataFrame({"o": [0.1, 0.2, 0.3, 0.4, 0.5], "bmi": [20.0, 21.0, np.nan, 23.0, 24.0]})
    out = pe.covariate_correlation(people, "o", "bmi")
    assert out["n"] == 4 and out["n_people_without_value"] == 1 and out["rho"] == pytest.approx(1.0)


def test_night_table_joins_results_and_participants():
    hotspots = {"per_night_rate": {"sub-1": 0.1, "sub-2": None}, "per_night_headband_missing_rate": {"sub-1": 0.0, "sub-2": 0.4}}
    clinical = {"per_night_metrics": {s: {"hum_sleep_efficiency": 80.0, "hum_sol": None, "hum_waso": 5.0, "hum_pct_rem": 20.0}
                                      for s in ("sub-1", "sub-2")}}
    participants = pd.DataFrame({"participant_id": ["sub-1", "sub-2"], "age": ["30", "n/a"], "sex": ["F", "M"],
                                 "bmi": ["21.5", "n/a"], "pid": ["1", "2"]})
    nights = pe.night_table(hotspots, clinical, participants)
    assert nights.loc["sub-1", "bmi"] == 21.5 and np.isnan(nights.loc["sub-2", "age"]) and nights.loc["sub-2", "order"] == 2
    assert np.isnan(nights.loc["sub-2", "disagreement_rate"]) and np.isnan(nights.loc["sub-1", "hum_sol"])
    assert nights.loc["sub-2", "pid"] == "2" and nights.loc["sub-2", "sex"] == "M"


def test_run_has_the_three_parts():
    rows = [(f"sub-{i}", str(i % 6), 0.01 * i, 0.1 + 0.01 * (i % 3), "F" if i % 2 else "M", 20.0 + i, 20.0 + i % 5, 80.0 + i % 7)
            for i in range(1, 25)]
    out = pe.run(make_nights(rows))
    assert set(out) >= {"a_person_or_night", "b_first_vs_later", "c_sex_bmi_age", "order_assumption"}
    assert out["a_person_or_night"]["n_people"] == 6 and set(out["b_first_vs_later"]) == set(pe.CHANGE_METRICS)
