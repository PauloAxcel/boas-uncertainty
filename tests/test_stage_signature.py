import numpy as np
import pandas as pd
import pytest
from test_signal_quality import synthetic_night

from boas_unc import edf
from boas_unc import stage_signature as ss


def test_relative_band_powers_of_a_flat_spectrum_follow_the_band_widths():
    freqs = np.arange(0.0, 50.0, 0.5)
    psd = np.ones((2, len(freqs)))
    out = ss.relative_band_powers(freqs, psd)
    assert out.shape == (2, 5) and out.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert out.delta[0] == pytest.approx(7 / 60)  # 0.5-4 Hz is 7 of the 60 bins between 0.5 and 30 Hz
    assert out.beta[0] == pytest.approx(29 / 60)  # 16-30 Hz includes the 30 Hz bin


def test_relative_band_powers_follow_the_frequency_of_the_signal_and_zero_power_is_nan():
    fs = 256.0
    t = np.arange(0, 30, 1 / fs)
    ep = np.stack([np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 10 * t), np.zeros_like(t)])
    from scipy import signal as sps
    freqs, psd = sps.welch(ep, fs=fs, nperseg=512, noverlap=0, axis=1)
    out = ss.relative_band_powers(freqs, psd)
    assert out.delta[0] > 0.9 and out.alpha[1] > 0.9 and out.iloc[2].isna().all()


def test_channel_table_combine_and_views(tmp_path):
    p = synthetic_night(tmp_path, n_epochs=3, flat_epoch=1)
    header = edf.read_header(p)
    views = ss.night_views(p, header)
    assert set(views) == {"HB_1", "HB_2", "combined"} and all(len(v) == 3 for v in views.values())
    assert views["HB_1"].iloc[1].isna().all()  # the flat epoch has no band powers and no log std
    assert views["HB_1"].iloc[0].delta == pytest.approx(views["combined"].iloc[0].delta, abs=0.2)


def test_combine_uses_the_other_channel_when_one_is_undefined():
    a = pd.DataFrame({"delta": [0.8, np.nan, np.nan]})
    b = pd.DataFrame({"delta": [0.6, 0.5, np.nan]})
    out = ss.combine([a, b])
    assert out.delta[0] == pytest.approx(0.7) and out.delta[1] == 0.5 and np.isnan(out.delta[2])


def test_stage_medians_needs_enough_epochs():
    view = pd.DataFrame({b: np.linspace(0.1, 0.5, 30) for b in ss.BANDS})
    stage = np.array([2] * 20 + [3] * 5 + [4] * 5)
    out = ss.stage_medians(view, stage)
    assert set(out) == {"N2"} and out["N2"]["n_epochs"] == 20


def test_contrast_auroc_direction_and_minimum_size():
    v = np.r_[np.full(25, 2.0), np.full(25, 1.0)]
    stage = np.r_[np.full(25, 3), np.full(25, 2)]
    assert ss.contrast_auroc(v, stage, (3,), (2,)) == 1.0
    assert ss.contrast_auroc(-v, stage, (3,), (2,)) == 0.0
    assert np.isnan(ss.contrast_auroc(v[:30], stage[:30], (3,), (2,)))  # only 5 epochs in the second class
    assert np.isnan(ss.contrast_auroc(np.full(50, np.nan), stage, (3,), (2,)))


def test_night_aurocs_has_every_contrast_and_feature():
    view = pd.DataFrame({**{b: np.random.default_rng(0).random(60) for b in ss.BANDS}, "log_std": np.random.default_rng(1).random(60)})
    out = ss.night_aurocs(view, np.repeat([0, 1, 2, 3, 4], 12))
    assert set(out) == set(ss.CONTRASTS) and set(out["N3_vs_N2"]) == set(ss.BANDS) | {"log_std"}


def test_spearman_guards():
    assert ss.spearman(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)
    assert np.isnan(ss.spearman(np.ones(5), np.arange(5.0))) and np.isnan(ss.spearman(np.arange(2.0), np.arange(2.0)))


def test_spike_epochs_are_relative_to_the_nights_median_of_live_epochs():
    std = np.array([10.0, 10.0, 10.0, 40.0, 0.0, 30.0])  # median of the live epochs is 10
    assert ss.spike_epochs(std).tolist() == [False, False, False, True, False, True]
    assert not ss.spike_epochs(np.zeros(4)).any()


def fake_night_result(n=60, seed=0):
    rng = np.random.default_rng(seed)
    stage = rng.choice([0, 1, 2, 3, 4], n)
    mk = lambda: pd.DataFrame({**{b: rng.random(n) for b in ss.BANDS}, "log_std": rng.random(n) + 1})
    views = {"HB_1": mk(), "HB_2": mk()}
    views[ss.COMBINED] = ss.combine([views["HB_1"], views["HB_2"]])
    std = 10 ** views["HB_1"].log_std.to_numpy()
    return ss.night_result(views, stage, std, rng.random(n)), stage


def test_night_result_has_every_part_and_the_movement_correlation():
    out, _ = fake_night_result()
    assert set(out["stage_medians"]) == set(ss.VIEWS) and set(out["aurocs"]) == set(ss.VIEWS)
    assert "spearman_log_std_vs_movement" in out and out["n_epochs_used"] == 60
    assert sum(out["live_by_stage"].values()) == 60 and sum(out["spikes_by_stage"].values()) == out["spike_epochs"]


def test_aggregate_summarises_over_nights_and_pools_the_spikes():
    nights = {f"sub-{i}": fake_night_result(seed=i)[0] for i in range(3)}
    out = ss.aggregate(nights)
    assert out["n_nights"] == 3 and set(out["stage_signature"]) == set(ss.VIEWS)
    assert out["stage_signature"]["combined"]["N2"]["n_nights"] == 3
    assert out["auroc"]["combined"]["N3_vs_N2"]["delta"]["n_defined"] <= 3
    assert sum(out["spikes"]["share_of_live_epochs_by_stage"].values()) == pytest.approx(1.0)
    assert out["spikes"]["n_live_epochs"] == 180 and out["channel_corr_log_std"]["n_defined"] == 3


def test_figure_is_written(tmp_path):
    nights = {f"sub-{i}": fake_night_result(seed=i)[0] for i in range(4)}
    result = {**ss.aggregate(nights), "per_night": nights}
    ss.fig_signature(result, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 1000
