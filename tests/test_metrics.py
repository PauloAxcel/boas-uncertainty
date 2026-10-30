import numpy as np
import pytest

from boas_unc import metrics as m

A = np.array
W, N1, N2, N3, R, X = 0, 1, 2, 3, 4, -2  # X = headband -2


def seq_of(*codes):
    """Sequence, 30 s onsets, and an all-valid mask."""
    seq = A(codes)
    return seq, np.arange(len(seq)) * 30, np.ones(len(seq), dtype=bool)


def identity_holds(r: dict) -> bool:
    return r["tib"] == pytest.approx(r["pre_sleep_wake"] + r["tst"] + r["waso"] + r["terminal_wake"] + r["unscored"])


def test_hand_built_10_wake_then_100_n2():
    seq, onset, valid = seq_of(*([W] * 10 + [N2] * 100))
    r = m.sleep_metrics(seq, onset, valid)
    assert r["tib"] == 55.0 and r["tst"] == 50.0
    assert r["sleep_efficiency"] == pytest.approx(100 * 50 / 55)
    assert r["sol"] == 5.0 and r["pre_sleep_wake"] == 5.0
    assert r["waso"] == 0.0 and r["terminal_wake"] == 0.0 and r["unscored"] == 0.0
    assert np.isnan(r["rem_latency"])
    assert r["pct_n2"] == 100.0 and r["pct_n1"] == 0.0
    assert identity_holds(r)


def test_hand_built_mixed_night():
    # idx: 0 W, 1 W, 2 N1, 3 N2, 4 W, 5 N2, 6 REM, 7 W, 8 W
    seq, onset, valid = seq_of(W, W, N1, N2, W, N2, R, W, W)
    r = m.sleep_metrics(seq, onset, valid)
    assert r["tib"] == 4.5 and r["tst"] == 2.0
    assert r["sol"] == 1.0 and r["pre_sleep_wake"] == 1.0
    assert r["waso"] == 0.5 and r["terminal_wake"] == 1.0
    assert r["waso_incl_terminal"] == 1.5
    assert r["rem_latency"] == 2.0  # REM at 180 s, onset at 60 s
    assert (r["pct_n1"], r["pct_n2"], r["pct_n3"], r["pct_rem"]) == (25.0, 50.0, 0.0, 25.0)
    assert identity_holds(r)


def test_unscored_epochs_count_in_tib_but_not_as_sleep_or_wake():
    seq, onset, valid = seq_of(W, X, N2, N2, X, W)
    r = m.sleep_metrics(seq, onset, valid)
    assert r["tib"] == 3.0 and r["unscored"] == 1.0 and r["tst"] == 1.0
    assert r["sol"] == 1.0 and r["pre_sleep_wake"] == 0.5  # SOL also contains the pre-onset -2
    assert r["waso"] == 0.0 and r["terminal_wake"] == 0.5
    assert identity_holds(r)


def test_excluded_epochs_leave_tib_but_stay_in_the_sol_timestamp():
    seq, onset, valid = seq_of(8, 8, W, N2, N2)
    valid[:2] = False
    r = m.sleep_metrics(seq, onset, valid)
    assert r["tib"] == 1.5 and r["pre_sleep_wake"] == 0.5
    assert r["sol"] == 1.5  # D8: timestamp difference, so the two excluded epochs count
    assert identity_holds(r)


def test_sleep_inside_an_excluded_stretch_is_neither_tst_nor_onset():
    seq, onset, valid = seq_of(W, N2, W, N2)
    valid[1] = False
    r = m.sleep_metrics(seq, onset, valid)
    assert r["tst"] == 0.5 and r["sol"] == 1.5 and r["tib"] == 1.5
    assert identity_holds(r)


def test_left_censored_night_has_no_sol_or_rem_latency():
    seq, onset, valid = seq_of(N2, N2, W, R, N2)
    r = m.sleep_metrics(seq, onset, valid)
    assert r["left_censored"] and np.isnan(r["sol"]) and np.isnan(r["rem_latency"])
    assert r["pre_sleep_wake"] == 0.0 and r["waso"] == 0.5
    assert identity_holds(r)


def test_left_censored_needs_a_valid_first_epoch():
    assert m.left_censored(A([N2, W]), A([True, True]))
    assert not m.left_censored(A([N2, W]), A([False, True]))
    assert not m.left_censored(A([W, N2]), A([True, True]))
    assert not m.left_censored(A([]), A([], dtype=bool))


def test_night_with_no_sleep_is_undefined_not_zero():
    seq, onset, valid = seq_of(W, W, W)
    r = m.sleep_metrics(seq, onset, valid)
    assert r["tst"] == 0.0 and r["sleep_efficiency"] == 0.0
    for key in ("sol", "pre_sleep_wake", "rem_latency", "waso", "terminal_wake", "pct_n2"):
        assert np.isnan(r[key]), key


def test_no_rem_gives_nan_rem_latency_only():
    seq, onset, valid = seq_of(W, N2, N2)
    r = m.sleep_metrics(seq, onset, valid)
    assert np.isnan(r["rem_latency"]) and r["sol"] == 0.5


def test_check_codes_rejects_unknown_and_nan_but_not_masked_codes():
    seq = A([W, 8, N2])
    with pytest.raises(ValueError):
        m.check_codes(seq, A([True, True, True]))
    m.check_codes(seq, A([True, False, True]))  # code 8 is fine once masked out
    with pytest.raises(ValueError):
        m.check_codes(A([W, np.nan]), A([True, True]))


def test_time_in_bed_and_total_sleep_time():
    seq, _, valid = seq_of(W, N1, X, N3)
    assert m.time_in_bed(valid) == 2.0 and m.total_sleep_time(seq, valid) == 1.0
    valid[3] = False
    assert m.time_in_bed(valid) == 1.5 and m.total_sleep_time(seq, valid) == 0.5


def test_sleep_efficiency():
    assert m.sleep_efficiency(50, 55) == pytest.approx(90.9090909)
    assert np.isnan(m.sleep_efficiency(0, 0))


def test_first_and_last_sleep_index():
    seq, _, valid = seq_of(W, X, N1, W, R, W)
    assert m.first_sleep_index(seq, valid) == 2 and m.last_sleep_index(seq, valid) == 4
    assert m.first_sleep_index(A([W, W]), A([True, True])) is None
    assert m.last_sleep_index(A([W, W]), A([True, True])) is None


def test_sleep_epochs_ignores_invalid():
    assert m.sleep_epochs(A([N2, N2, W]), A([True, False, True])).tolist() == [True, False, False]


def test_stage_percentages_sum_to_100():
    seq, _, valid = seq_of(N1, N2, N2, N3, R, W, X)
    pct = m.stage_percentages(seq, valid)
    assert sum(pct.values()) == pytest.approx(100.0)
    assert pct["pct_n2"] == pytest.approx(40.0)


def test_stage_percentages_nan_without_sleep():
    assert all(np.isnan(v) for v in m.stage_percentages(A([W, X]), A([True, True])).values())


def test_pre_sleep_wake_waso_terminal_wake_separately():
    seq, _, valid = seq_of(W, W, N2, W, W, N2, W)
    assert m.pre_sleep_wake(seq, valid) == 1.0
    assert m.waso(seq, valid) == 1.0
    assert m.terminal_wake(seq, valid) == 0.5


def test_unscored_time_respects_valid():
    seq, _, valid = seq_of(X, X, N2)
    valid[0] = False
    assert m.unscored_time(seq, valid) == 0.5


def test_first_run_start():
    assert m.first_run_start(A([False, True, True, False, True, True, True]), 3) == 4
    assert m.first_run_start(A([True, True, False, True]), 3) is None
    assert m.first_run_start(A([True, True, True]), 3) == 0
    assert m.first_run_start(A([], dtype=bool), 1) is None


def test_sustained_rem_latency_skips_short_rem_stretches():
    # idx: 0 W, 1 N2, 2 REM (1 epoch), 3 N2, 4 REM, 5 REM, 6 REM, 7 N2
    seq, onset, valid = seq_of(W, N2, R, N2, R, R, R, N2)
    assert m.rem_latency(seq, onset, valid) == 0.5  # first REM epoch: idx 2 at 60 s, onset idx 1 at 30 s
    assert m.sustained_rem_latency(seq, onset, valid) == 1.5  # run starts at idx 4 (120 s)
    assert m.sustained_rem_latency(seq, onset, valid, min_run=1) == m.rem_latency(seq, onset, valid)


def test_sustained_rem_latency_is_nan_without_a_long_enough_run_or_when_censored():
    seq, onset, valid = seq_of(W, N2, R, R, N2)
    assert m.rem_latency(seq, onset, valid) == 0.5 and np.isnan(m.sustained_rem_latency(seq, onset, valid))
    cens, onset2, valid2 = seq_of(N2, R, R, R)
    assert np.isnan(m.sustained_rem_latency(cens, onset2, valid2))


def test_excluded_epoch_breaks_a_rem_run():
    seq, onset, valid = seq_of(W, N2, R, R, R, R, R)
    valid[3] = False  # REM, excluded, REM, REM, REM: the run after the gap is 3 long and starts at idx 4
    assert m.sustained_rem_latency(seq, onset, valid) == 1.5


def test_sleep_metrics_includes_sustained_rem_latency():
    seq, onset, valid = seq_of(W, N2, R, R, R)
    assert m.sleep_metrics(seq, onset, valid)["rem_latency_sustained"] == 0.5


def test_matched_mask_and_matched_metrics():
    hum = A([W, N2, N2, 8, N2, N2])
    hb = A([W, N2, X, N2, N2, N3])
    valid = hum != 8
    matched = m.matched_mask(hum, hb, valid)
    assert matched.tolist() == [True, True, False, False, True, True]
    r = m.matched_metrics(hb, matched)
    assert r["tib_matched"] == 2.0 and r["tst"] == 1.5
    assert r["sleep_efficiency"] == pytest.approx(75.0)
    assert r["pct_n2"] == pytest.approx(200 / 3) and r["pct_n3"] == pytest.approx(100 / 3)
    assert "sol" not in r and "waso" not in r  # no latencies or WASO in the matched analysis


def test_sleep_metrics_equals_matched_metrics_when_nothing_is_dropped():
    seq, onset, valid = seq_of(W, N2, N2, R, W)
    full, matched = m.sleep_metrics(seq, onset, valid), m.matched_metrics(seq, m.matched_mask(seq, seq, valid))
    for key in ("tst", "sleep_efficiency", "pct_n2", "pct_rem"):
        assert full[key] == matched[key]
