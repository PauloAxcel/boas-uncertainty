import numpy as np
import pandas as pd
import pytest

from boas_unc import one_channel as oc


def test_reference_amplitude_uses_staged_epochs_with_signal():
    std = np.array([10.0, 20.0, 0.0, 1000.0, 30.0])
    staged = np.array([True, True, True, False, True])
    assert oc.reference_amplitude(std, staged) == 20.0  # median of 10, 20 and 30; the zero and the unstaged epoch are out
    assert np.isnan(oc.reference_amplitude(std, np.zeros(5, dtype=bool)))


def test_bad_channel_by_flatness_high_and_low_amplitude():
    flat = np.array([0.0, 0.0, 0.0, 0.6, 0.0])
    std = np.array([20.0, 500.0, 1.0, 20.0, 200.0])
    out = oc.bad_channel(flat, std, ref=20.0, high=10.0, low=0.1)
    assert out.tolist() == [False, True, True, True, True]  # 500 and 200 are 25x and 10x, 1.0 is 0.05x, flat share 0.6
    assert oc.bad_channel(np.zeros(1), np.array([100.0]), 20.0, high=10.0, low=0.1).tolist() == [False]
    assert oc.bad_channel(np.zeros(1), np.array([100.0]), 20.0, high=5.0, low=0.2).tolist() == [True]  # sensitivity


def test_epoch_status_counts_bad_channels():
    assert oc.epoch_status(np.array([False, True, False, True]), np.array([False, False, True, True])).tolist() == [0, 1, 1, 2]


def make_night():
    """8 epochs: 6 staged and clean, then 1 no-stage epoch with HB_2 wild, then 1 no-stage epoch with both flat."""
    n = 8
    d = {"missing": [False] * 6 + [True, True], "staged": [True] * 6 + [False, False]}
    for ch, wild in (("HB_1", 20.0), ("HB_2", 900.0)):
        d[f"{ch}_std_uv"] = [20.0] * 6 + [wild if ch == "HB_2" else 20.0, 0.0]
        d[f"{ch}_flat_share"] = [0.0] * 6 + [0.0, 1.0]
    return pd.DataFrame(d), np.array([2, 2, 2, 3, 3, 3, 2, 2]), np.array([2, 2, 3, 3, 3, 2, -2, -2])


def test_night_counts_classifies_the_epochs_and_counts_agreement():
    joined, hum, hb = make_night()
    out = oc.night_counts(joined, hum, hb, 10.0, 0.1)
    assert out["missing_by_status"] == {"both_good": 0, "one_good": 1, "both_bad": 1}
    assert out["staged_by_status"] == {"both_good": 6, "one_good": 0, "both_bad": 0}
    assert out["one_good_missing_bad_channel"] == {"HB_1": 0, "HB_2": 1}
    assert out["agreement_by_status"]["both_good"] == {"n": 6, "n_agree": 4}
    assert out["recoverable_share_of_missing"] == 0.5 and out["recoverable_minutes"] == 0.5


def test_night_counts_is_none_without_a_reference():
    joined, hum, hb = make_night()
    joined["staged"] = False
    assert oc.night_counts(joined, hum, hb, 10.0, 0.1) is None


def test_aggregate_pools_nights_and_ignores_skipped_ones():
    joined, hum, hb = make_night()
    one = oc.night_counts(joined, hum, hb, 10.0, 0.1)
    out = oc.aggregate({"a": one, "b": one, "c": None})
    assert out["n_nights"] == 2 and out["pooled"]["missing_by_status"]["one_good"] == 2
    assert out["share_of_missing_by_status"]["one_good"] == 0.5
    assert out["share_of_one_good_epochs_without_a_stage"] == 1.0  # both one-good epochs have no stage
    assert out["agreement"]["both_good"]["accuracy"] == pytest.approx(4 / 6)
    assert out["pooled_recoverable_minutes"] == 1.0 and out["pooled_missing_minutes"] == 2.0
    assert out["recoverable_share_per_night"]["n_defined"] == 2
