from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from test_signal_quality import synthetic_night

from boas_unc import edf
from boas_unc import plot_night as pn
from boas_unc import signal_quality as sq


def test_bandpass_removes_dc_and_keeps_alpha_band():
    fs = 256.0
    t = np.arange(0, 20, 1 / fs)
    x = -3000.0 + 20 * np.sin(2 * np.pi * 10 * t)
    y = pn.bandpass(x, fs)
    mid = y[int(5 * fs):int(15 * fs)]
    assert abs(mid.mean()) < 1 and mid.std() == pytest.approx(20 / np.sqrt(2), rel=0.1)


def test_bandpass_attenuates_slow_drift_and_fast_noise():
    fs = 256.0
    t = np.arange(0, 30, 1 / fs)
    slow, fast = 100 * np.sin(2 * np.pi * 0.05 * t), 100 * np.sin(2 * np.pi * 100 * t)
    mid = slice(int(8 * fs), int(22 * fs))
    assert pn.bandpass(slow, fs)[mid].std() < 5 and pn.bandpass(fast, fs)[mid].std() < 1


def test_window_trace_returns_the_window_times_and_raw_level():
    fs = 100.0
    x = np.concatenate([np.zeros(3000), np.full(3000, -2999.9)])
    seg, t, level = pn.window_trace(x, fs, start_s=40.0, seconds=10.0)
    assert len(seg) == 1000 and t[0] == 40.0 and t[-1] == pytest.approx(49.99)
    assert level == -2999.9
    seg0, t0, level0 = pn.window_trace(x, fs, start_s=0.0, seconds=5.0)  # a window at the very start is still fine
    assert len(seg0) == 500 and level0 == 0.0


def test_pick_windows_finds_n2_and_the_first_long_dropout():
    hb = np.array([0] * 3 + [2] * 20 + [3] * 5 + [-2] * 12 + [2] * 5)
    windows = dict(pn.pick_windows(hb, 60.0))
    assert windows["headband N2"] == pytest.approx((3 + 10) * 30)  # middle of the N2 run
    assert windows["start of the first no-stage stretch"] == (3 + 20 + 5 - 1) * 30


def test_pick_windows_uses_a_short_gap_when_there_is_no_long_run_and_handles_nothing_to_show():
    short = np.array([2] * 15 + [-2] * 3 + [2] * 15)
    assert pn.pick_windows(short, 60.0)[-1][1] == (15 - 1) * 30
    assert pn.pick_windows(np.array([0, 1, 0, 1, 0]), 60.0) == []


def test_stage_text_names_the_epochs_of_a_window():
    night = pd.DataFrame({"psg_hum": [0, 2, 8, 4], "hb_ai": [0, 2, -2, 4]})
    text = pn.stage_text(night, start_s=30.0, seconds=60.0)
    assert "epochs 1-2" in text and "human: N2 8" in text and "headband: N2 -2" in text


def test_overview_and_zoom_figures_are_written(tmp_path):
    p = synthetic_night(tmp_path, n_epochs=3, flat_epoch=2)
    header = edf.read_header(p)
    night = pd.DataFrame({"psg_hum": [2, 2, 2], "hb_ai": [2, 2, -2]})
    joined = sq.align(night, sq.night_quality(p, header))
    out1, out2 = tmp_path / "o.png", tmp_path / "z.png"
    pn.fig_overview(night, joined, None, header, "sub-x", out1)
    pn.fig_overview(night, joined, np.array([0.1, 0.2, 5.0]), header, "sub-x", tmp_path / "o2.png")  # with movement
    pn.fig_zoom(night, p, header, "sub-x", [("a window", 0.0), ("another", 30.0)], 30.0, out2)
    assert out1.stat().st_size > 1000 and out2.stat().st_size > 1000 and (tmp_path / "o2.png").exists()


def test_plot_night_refuses_an_incomplete_file(tmp_path):
    with pytest.raises((SystemExit, FileNotFoundError, OSError)):
        pn.plot_night("sub-1", tmp_path, tmp_path, 60.0, None)


def test_command_line_needs_a_recording_or_all():
    with pytest.raises(SystemExit):
        pn.main([])


def test_epoch_psd_peaks_at_the_signal_frequency_and_stays_in_the_band():
    fs = 256.0
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t)
    freqs, psd = pn.epoch_psd(x, fs)
    assert psd.shape == (2, len(freqs)) and freqs.min() >= pn.SPEC_BAND[0] and freqs.max() <= pn.SPEC_BAND[1]
    assert freqs[np.argmax(psd[0])] == pytest.approx(10.0, abs=0.5)


def test_to_db_turns_zero_power_into_nan():
    out = pn.to_db(np.array([[100.0, 0.0], [1.0, 0.01]]))
    assert out[0, 0] == pytest.approx(20.0) and np.isnan(out[0, 1])
    assert out[1].tolist() == pytest.approx([0.0, -20.0])


def test_pick_windows_unlabelled_finds_a_typical_epoch_and_the_first_flat_stretch():
    flat = np.array([False] * 5 + [True] * 12 + [False] * 3)
    std = np.array([10.0, 20.0, 30.0, 40.0, 50.0] + [0.0] * 12 + [20.0, 30.0, 25.0])
    windows = dict(pn.pick_windows_unlabelled(flat, std))
    assert windows["a typical epoch"] == 2 * 30  # median of the live epochs is 25 (epoch 2 has 30, epoch 19 has 25)
    assert windows["start of the first flat stretch"] == (5 - 1) * 30


def test_pick_windows_unlabelled_handles_no_flat_and_all_flat():
    assert [w[0] for w in pn.pick_windows_unlabelled(np.zeros(10, dtype=bool), np.arange(10.0))] == ["a typical epoch"]
    assert [w[0] for w in pn.pick_windows_unlabelled(np.ones(10, dtype=bool), np.zeros(10))] == [
        "start of the first flat stretch"]


def test_overview_with_a_spectrogram_and_without_labels(tmp_path):
    p = synthetic_night(tmp_path, n_epochs=3, flat_epoch=1)
    header = edf.read_header(p)
    quality = sq.night_quality(p, header)
    spectrum = pn.spectrum_of(p, header)
    assert spectrum[1].shape[0] == 3
    bands = pn.bands_of(p, header)
    assert list(bands.columns)[:5] == list(pn.ss.BANDS) and len(bands) == 3
    out = tmp_path / "o.png"
    pn.fig_overview(None, quality, None, header, "my_night", out, spectrum, bands)
    assert out.stat().st_size > 1000


def test_plot_file_writes_both_figures_without_labels(tmp_path):
    p = synthetic_night(tmp_path, n_epochs=3, flat_epoch=1)
    paths = pn.plot_file(p, tmp_path / "figs", 30.0, None)
    assert [q.name for q in paths] == ["n_overview.png", "n_zoom.png"] and all(q.stat().st_size > 1000 for q in paths)
    assert pn.plot_file(p, tmp_path / "figs2", 30.0, 0.005)[1].exists()  # a window chosen by hand


def test_plot_file_asks_for_the_channel_names(tmp_path):
    from test_edf import write_edf
    p = tmp_path / "odd.edf"
    write_edf(p, [dict(label="EEG_A", samples=100, pmin=-3000.0, pmax=3000.0, dmin=-32768, dmax=32767)], 30,
              data=np.zeros((30, 100)))
    with pytest.raises(SystemExit, match="HB_1"):
        pn.plot_file(p, tmp_path / "figs", 30.0, None)


def test_command_line_file_mode(tmp_path, capsys):
    p = synthetic_night(tmp_path, n_epochs=3, flat_epoch=1)
    pn.main(["--file", str(p), "--out", str(tmp_path / "cli")])
    assert "n_overview.png" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pn.main(["--file", str(p), "sub-1"])  # a file and a recording together are refused
