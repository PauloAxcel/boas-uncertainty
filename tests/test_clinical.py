import numpy as np
import pandas as pd
import pytest

from boas_unc import clinical as cl

A = np.array
S = pd.Series


def test_ba_stats_constant_difference():
    out = cl.ba_stats(A([1.0, 2.0, 3.0]), A([2.0, 3.0, 4.0]))  # headband is always 1 higher
    assert out["n"] == 3 and out["bias"] == 1.0 and out["sd_difference"] == 0.0
    assert out["loa_lower"] == out["loa_upper"] == 1.0
    assert out["pearson_r"] == pytest.approx(1.0) and out["spearman_rho"] == pytest.approx(1.0)


def test_ba_stats_limits_use_1_96_sd_of_the_differences():
    human, headband = A([10.0, 20.0, 30.0, 40.0]), A([12.0, 17.0, 33.0, 38.0])  # d = 2, -3, 3, -2
    out = cl.ba_stats(human, headband)
    sd = np.std([2, -3, 3, -2], ddof=1)
    assert out["bias"] == 0.0 and out["sd_difference"] == pytest.approx(sd)
    assert out["loa_upper"] == pytest.approx(1.96 * sd) and out["loa_lower"] == pytest.approx(-1.96 * sd)


def test_ba_stats_correlation_undefined_when_too_few_or_constant():
    assert np.isnan(cl.ba_stats(A([1.0, 2.0]), A([1.0, 2.0]))["pearson_r"])
    assert np.isnan(cl.ba_stats(A([1.0, 1.0, 1.0]), A([1.0, 2.0, 3.0]))["spearman_rho"])
    assert np.isnan(cl.ba_stats(A([]), A([]))["bias"])


def test_paired_keeps_only_nights_with_both_values():
    h, b = cl.paired(S({"a": 1.0, "b": np.nan, "c": 3.0}), S({"a": 1.0, "b": 2.0, "c": np.nan}))
    assert h.index.tolist() == ["a"] and b.index.tolist() == ["a"]


def test_bland_altman_reports_who_is_excluded():
    human = S({"a": 1.0, "b": np.nan, "c": 3.0, "d": np.nan, "e": 5.0})
    headband = S({"a": 2.0, "b": 2.0, "c": np.nan, "d": np.nan, "e": 6.0})
    out = cl.bland_altman(human, headband)
    assert out["n_nights_used"] == 2
    assert out["excluded"] == {"human_undefined": 1, "headband_undefined": 1, "both_undefined": 1}
    assert out["bias"] == 1.0 and out["human"]["n_undefined"] == 2


def test_bland_altman_by_person_averages_a_persons_nights():
    human = S({"n1": 10.0, "n2": 20.0, "n3": 30.0})
    headband = S({"n1": 11.0, "n2": 21.0, "n3": 40.0})
    pid = S({"n1": "p", "n2": "p", "n3": "q"})
    out = cl.bland_altman(human, headband, pid)
    assert out["n"] == 3 and out["by_person"]["n"] == 2  # people, not nights
    assert out["by_person"]["bias"] == pytest.approx((1.0 + 10.0) / 2)


def make_night(hum, hb):
    n = len(hum)
    return pd.DataFrame({"epoch": range(n), "onset": np.arange(n) * 30, "psg_hum": hum, "hb_ai": hb,
                         "hum_disconnected": A(hum) == 8})


def test_night_row_has_both_sequences_and_matched_metrics():
    row = cl.night_row(make_night([0, 0, 2, 2, 2, 4], [0, 0, 2, -2, 2, 4]))
    assert row["hum_tst"] == 2.0 and row["hb_tst"] == 1.5
    assert row["hb_unscored"] == 0.5 and row["hum_unscored"] == 0.0
    assert row["hb_dropout_share"] == pytest.approx(1 / 6)
    assert row["mhum_tib_matched"] == 2.5 and row["mhum_tst"] == 1.5  # the -2 epoch is not matched


def test_night_row_flags_dropout_before_the_human_targets():
    row = cl.night_row(make_night([0, 2, 2, 4], [-2, 2, 2, 4]))
    assert row["flag_sol_dropout"] and row["flag_rem_dropout"]
    clean = cl.night_row(make_night([0, 2, 2, 4], [0, 2, -2, 4]))
    assert not clean["flag_sol_dropout"] and clean["flag_rem_dropout"]  # the -2 comes before the REM epoch


def test_night_row_excludes_human_8_unless_treated_as_wake():
    night = make_night([8, 8, 0, 2, 2], [0, 0, 0, 2, 2])
    assert cl.night_row(night)["hum_tib"] == 1.5
    as_wake = cl.night_row(night, treat_8_as_wake=True)
    assert as_wake["hum_tib"] == 2.5 and as_wake["hb_tib"] == 2.5
    assert as_wake["hum_pre_sleep_wake"] == 1.5  # the two 8 epochs and the wake epoch


def test_all_nights_and_identity_error_are_zero_on_clean_nights():
    t = pd.concat([make_night([0, 2, 2, 8, 0], [0, 2, -2, 0, 0]).assign(sub="a"),
                   make_night([0, 0, 2, 4, 2], [0, 2, 2, 4, 2]).assign(sub="b")], ignore_index=True)
    nights = cl.all_nights(t)
    assert nights.index.tolist() == ["a", "b"] and cl.identity_error(nights) == 0.0


def test_zero_floor_counts_split_the_nights_by_which_side_is_zero():
    nights = pd.DataFrame({"hum_x": [0.0, 0.0, 5.0, 3.0, np.nan], "hb_x": [0.0, 2.0, 0.0, 3.0, 1.0]})
    assert cl.zero_floor_counts(nights, "x") == {"n": 4, "both_zero": 1, "headband_zero_human_above": 1,
                                                 "human_zero_headband_above": 1}


def test_metric_block_covers_every_metric():
    t = pd.concat([make_night([0, 2, 2, 4, 2], [0, 2, 2, 4, 2]).assign(sub=s) for s in ("a", "b", "c")], ignore_index=True)
    out = cl.metric_block(cl.all_nights(t), ["tst", "waso"], S({"a": "1", "b": "2", "c": "3"}))
    assert set(out) == {"tst", "waso"} and out["tst"]["bias"] == 0.0
