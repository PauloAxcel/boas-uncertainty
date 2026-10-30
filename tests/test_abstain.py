import numpy as np
import pandas as pd
import pytest

from boas_unc import abstain as ab

A = np.array
W, N1, N2, N3, R, X = 0, 1, 2, 3, 4, -2


def test_is_stage():
    assert ab.is_stage(A([0, 4, 8, -2, np.nan])).tolist() == [True, True, False, False, False]


def test_change_flag_marks_both_sides_and_widens_with_k():
    hb = A([N2, N2, N3, N3, N3, N3])
    assert ab.change_flag(hb, 0).tolist() == [False, True, True, False, False, False]
    assert ab.change_flag(hb, 1).tolist() == [True, True, True, True, False, False]
    assert ab.change_flag(hb, 2).tolist() == [True, True, True, True, True, False]


def test_change_flag_ignores_boundaries_with_minus2():
    assert not ab.change_flag(A([N2, N2, X, X, N3, N3]), 0).any()
    assert not ab.change_flag(A([N2, N2, N2]), 5).any()  # no change anywhere


def test_isolated_flag_needs_two_different_stage_neighbours():
    assert ab.isolated_flag(A([N2, N2, N3, N2, N2])).tolist() == [False, False, True, False, False]
    assert ab.isolated_flag(A([N2, W, N3, N3])).tolist() == [False, True, False, False]  # neighbours differ from each other
    assert not ab.isolated_flag(A([N3, N2, N2])).any()  # the first epoch is never flagged
    assert not ab.isolated_flag(A([N2, X, N3, N2, N2])).any()  # a -2 neighbour is not a stage
    assert not ab.isolated_flag(A([N2, N3])).any()


def test_isolated_epochs_are_always_change_epochs():
    hb = A([N2, W, N2, N3, R, R, N1, N2, N2])
    assert (ab.isolated_flag(hb) <= ab.change_flag(hb, 0)).all()


def test_artefact_flag_excludes_the_minus2_epochs_themselves():
    hb = A([N2, N2, X, N2, N2, N2])
    assert ab.artefact_flag(hb, 1).tolist() == [False, True, False, True, False, False]
    assert ab.artefact_flag(hb, 2).tolist() == [True, True, False, True, True, False]
    assert not ab.artefact_flag(A([N2, N2]), 3).any()


def test_device_flags_names_and_nesting():
    hb = A([N2, N2, N3, N2, N2, X, N2, N2, N2, N2])
    flags = ab.device_flags(hb)
    assert not flags["none"].any()
    assert {"isolated", "change_0", "change_5", "artefact_1", "artefact_5", "combined_0", "combined_5"} <= set(flags)
    assert (flags["change_0"] <= flags["change_1"]).all() and (flags["artefact_1"] <= flags["artefact_2"]).all()
    assert (flags["combined_2"] == (flags["change_2"] | flags["artefact_2"])).all()
    assert (flags["combined_0"] == flags["change_0"]).all()


def test_psg_disagree_flag_counts_a_psg_minus2_as_a_difference():
    assert ab.psg_disagree_flag(A([N2, N2, X]), A([N2, N3, N2])).tolist() == [False, True, True]


def test_smooth_isolated_only_changes_epochs_with_equal_neighbours():
    assert ab.smooth_isolated(A([N2, N2, N3, N2, N2])).tolist() == [N2] * 5
    assert ab.smooth_isolated(A([N2, W, N3, N3])).tolist() == [N2, W, N3, N3]  # neighbours differ: unchanged
    assert ab.smooth_isolated(A([N3, N2, N2])).tolist() == [N3, N2, N2]
    assert ab.smooth_isolated(A([N2])).tolist() == [N2]
    hb = A([N2, N2, N3, N2, N2])
    ab.smooth_isolated(hb)
    assert hb.tolist() == [N2, N2, N3, N2, N2]  # the input is not modified


def test_family():
    assert [ab.family(n) for n in ("change_3", "artefact_1", "combined_0", "isolated", "none", ab.PSG_FLAG)] == \
        ["change", "artefact", "combined", "isolated", "none", ab.PSG_FLAG]


def make_table(hum, hb, psg_ai=None, sub="s1"):
    n = len(hum)
    return pd.DataFrame({"sub": sub, "epoch": range(n), "onset": np.arange(n) * 30, "psg_hum": hum,
                         "psg_ai": hum if psg_ai is None else psg_ai, "hb_ai": hb, "hum_disconnected": A(hum) == 8})


def test_add_flags_adds_one_column_per_flag_set_and_the_smoothed_label():
    df = ab.add_flags(make_table([N2] * 5, [N2, N2, N3, N2, N2]))
    assert "flag_isolated" in df and f"flag_{ab.PSG_FLAG}" in df and "flag_none" in df
    assert df.hb_smooth.tolist() == [N2] * 5
    assert set(ab.flag_names(df)) >= {"none", "change_0", "artefact_1", "combined_3", ab.PSG_FLAG}


def test_flag_summary_hand_computed():
    df = pd.DataFrame({"sub": "s", "psg_hum": [0, 1, 2, 3, 8], "hb_ai": [0, 2, 2, -2, 0]})
    base = ab.both_stage(df)  # epochs 0, 1, 2: epoch 3 is -2, epoch 4 is human 8
    flag = pd.Series([False, True, False, False, False])
    out = ab.flag_summary(df, flag, base, n_human=4)
    assert (out["n_base"], out["n_kept"], out["n_flagged"]) == (3, 2, 1)
    assert out["coverage"] == pytest.approx(2 / 3) and out["coverage_of_human_epochs"] == 0.5
    assert out["accuracy_kept"] == 1.0 and out["accuracy_flagged"] == 0.0 and out["error_capture"] == 1.0


def test_flag_summary_no_flag_is_the_baseline():
    df = pd.DataFrame({"sub": "s", "psg_hum": [0, 1, 2, 2], "hb_ai": [0, 2, 2, 2]})
    out = ab.flag_summary(df, pd.Series([False] * 4), ab.both_stage(df), n_human=4)
    assert out["coverage"] == 1.0 and out["accuracy_kept"] == 0.75 and np.isnan(out["accuracy_flagged"])


def test_per_night_flag_needs_enough_kept_epochs():
    big = make_table([N2] * 12, [N2] * 11 + [N3], sub="big")
    small = make_table([N2] * 4, [N2] * 4, sub="small")
    df = pd.concat([big, small], ignore_index=True)
    out = ab.per_night_flag(df, pd.Series(False, index=df.index), ab.both_stage(df))
    assert out["coverage"]["n_defined"] == 2 and out["coverage"]["median"] == 1.0
    assert out["accuracy_kept"]["n_defined"] == 1 and out["accuracy_kept"]["n_undefined"] == 1
    assert out["accuracy_kept"]["median"] == pytest.approx(11 / 12)


def test_evaluate_flags_none_equals_baseline_and_flagging_errors_helps():
    hum = [N2] * 12 + [N3] * 12
    hb = [N2] * 11 + [N3] + [N3] * 11 + [N2]  # one error at each side of the single change
    df = ab.add_flags(make_table(hum, hb))
    out = ab.evaluate_flags(df)
    assert out["none"]["coverage"] == 1.0 and out["none"]["accuracy_kept"] == pytest.approx(22 / 24)
    assert out["change_1"]["accuracy_kept"] > out["none"]["accuracy_kept"]
    assert out["change_1"]["coverage"] < 1.0 and out[ab.PSG_FLAG]["on_device"] is False


def test_error_summary_and_compare_errors():
    out = ab.error_summary(pd.Series({"a": 10.0, "b": 20.0, "c": 30.0}), pd.Series({"a": 12.0, "b": 17.0, "c": np.nan}))
    assert out["n"] == 2 and out["bias"] == pytest.approx(-0.5) and out["median_abs_error"] == pytest.approx(2.5)
    assert out["loa_width"] == pytest.approx(2 * 1.96 * out["sd_difference"])
    before = {"bias": -4.0, "loa_width": 10.0, "median_abs_error": 3.0}
    after = {"bias": 1.0, "loa_width": 8.0, "median_abs_error": 3.5}
    cmp = ab.compare_errors(before, after)
    assert cmp["change_in_abs_bias"] == -3.0 and cmp["change_in_loa_width"] == -2.0
    assert cmp["change_in_median_abs_error"] == 0.5


def test_excluded_night_metrics_drops_flagged_epochs_from_both_sequences():
    night = make_table([W, N2, N2, N2, N2], [W, N2, N2, N3, N2])
    full = ab.excluded_night_metrics(night, None)
    assert full["hum_tib_matched"] == 2.5 and full["hb_pct_n3"] == pytest.approx(25.0)
    kept = ab.excluded_night_metrics(night, A([False, False, False, True, False]))
    assert kept["hum_tib_matched"] == 2.0 and kept["hb_pct_n3"] == 0.0


def test_run_on_a_small_table_has_all_sections():
    rng = np.random.default_rng(0)
    parts = []
    for sub in ("a", "b", "c"):
        hum = np.concatenate([[W] * 4, rng.choice([N2, N3, R], 40, p=[0.7, 0.2, 0.1]), [W] * 2])
        hb = hum.copy()
        hb[rng.integers(5, 40, 4)] = N1
        parts.append(make_table(hum, hb, sub=sub))
    out = ab.run(pd.concat(parts, ignore_index=True), baseline_accuracy_from_agreement=None)
    assert out["baseline_matches_agreement_json"] is None
    assert out["flag_sets"]["none"]["coverage"] == 1.0
    assert out["flag_sets"]["none"]["accuracy_kept"] == pytest.approx(out["baseline_accuracy"])
    assert set(out["clinical_excluded"]) == set(ab.CLINICAL_OPERATING_POINTS)
    assert {"sol", "waso", "rem_latency"} <= set(out["clinical_smoothed"]["clinical"])
    assert out["clinical_smoothed"]["agreement_version_a"]["before"]["n_epochs"] == 3 * 46
