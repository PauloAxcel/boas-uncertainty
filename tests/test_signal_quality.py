import hashlib
import os

import numpy as np
import pandas as pd
import pytest
from test_edf import write_edf

from boas_unc import signal_quality as sq

FS = 100.0


def test_annex_key_parses_size_and_checksum(tmp_path):
    sha = "ab" * 32
    eeg = tmp_path / "sub-1" / "eeg"
    eeg.mkdir(parents=True)
    os.symlink(f"../../.git/annex/objects/x/y/SHA256E-s1234--{sha}.edf/SHA256E-s1234--{sha}.edf",
               eeg / "sub-1_task-Sleep_acq-headband_eeg.edf")
    assert sq.annex_key(tmp_path, "sub-1") == (1234, sha)


def test_annex_key_rejects_a_link_without_a_key(tmp_path):
    eeg = tmp_path / "sub-1" / "eeg"
    eeg.mkdir(parents=True)
    os.symlink("somewhere/else.edf", eeg / "sub-1_task-Sleep_acq-headband_eeg.edf")
    with pytest.raises(ValueError):
        sq.annex_key(tmp_path, "sub-1")


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 1000)
    assert sq.sha256_file(p, chunk=7) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_is_complete_checks_existence_size_and_checksum(tmp_path):
    p = tmp_path / "f.edf"
    assert not sq.is_complete(p, (3, "x"))  # missing
    p.write_bytes(b"abc")
    good = (3, hashlib.sha256(b"abc").hexdigest())
    assert sq.is_complete(p, good)
    assert not sq.is_complete(p, (4, good[1]))  # wrong size: a partial download
    assert not sq.is_complete(p, (3, "0" * 64))  # same size, wrong content


def test_complete_nights_separates_finished_and_unfinished_files(tmp_path):
    data, edf_dir = tmp_path / "data", tmp_path / "edfs"
    for sub, content in (("sub-1", b"abc"), ("sub-2", b"ab")):
        (data / sub / "eeg").mkdir(parents=True)
        sha = hashlib.sha256(b"abc").hexdigest()  # both nights expect the 3-byte content
        os.symlink(f"x/SHA256E-s3--{sha}.edf", data / sub / "eeg" / f"{sub}_task-Sleep_acq-headband_eeg.edf")
        target = sq.edf_path(edf_dir, sub)
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
    ok, skipped = sq.complete_nights(data, edf_dir)
    assert ok == ["sub-1"] and skipped == ["sub-2"]


def test_epoch_features_known_answers():
    ep = np.zeros((3, 100))
    ep[1] = np.arange(100.0)                       # a ramp: never flat
    ep[2] = 3000.0                                 # stuck at the range limit
    f = sq.epoch_features(ep, FS, limit_uv=3000.0)
    assert f.flat_share.tolist() == [1.0, 0.0, 1.0]
    assert f.sat_share.tolist() == [0.0, 0.0, 1.0] or f.sat_share.tolist() == [0.0, 0.01, 1.0]
    assert f.std_uv[0] == 0 and f.p2p_uv[1] == 99
    assert np.isnan(f.hf_ratio[0]) and np.isnan(f.mains_ratio[2])  # no power: undefined, not zero


def test_a_constant_epoch_at_a_non_zero_level_is_exactly_flat_and_has_no_spectrum():
    ep = np.full((2, 7680), -2999.9)
    ep[1] += np.random.default_rng(0).normal(0, 20, 7680)
    f = sq.epoch_features(ep, 256.0, limit_uv=3000.0)
    assert f.std_uv[0] == 0.0 and f.std_uv[1] > 10  # not 1e-13 from rounding
    assert np.isnan(f.hf_ratio[0]) and np.isnan(f.mains_ratio[0]) and not np.isnan(f.hf_ratio[1])
    assert sq.constant_epochs(ep).tolist() == [True, False]


def test_spectral_ratios_follow_the_frequency_of_the_signal():
    t = np.arange(0, 30, 1 / 256.0)
    ep = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 40 * t),
                   np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)])
    hf, mains = sq.spectral_ratios(ep, 256.0)
    assert hf[0] < 0.01 and hf[1] > 0.9
    assert mains[0] < 0.01 and mains[2] == pytest.approx(1.0, rel=0.3)


def synthetic_night(tmp_path, n_epochs=2, flat_epoch=None):
    n_records = int(n_epochs * 30)
    sigs = [dict(label=ch, samples=int(FS), pmin=-3000.0, pmax=3000.0, dmin=-32768, dmax=32767) for ch in sq.CHANNELS]
    rng = np.random.default_rng(0)
    data = []
    for r in range(n_records):
        rows = []
        for _ in sq.CHANNELS:
            v = rng.integers(-500, 500, int(FS))
            if flat_epoch is not None and r // 30 == flat_epoch:
                v = np.full(int(FS), -32767)
            rows.append(v)
        data.append(np.concatenate(rows))
    p = tmp_path / "n.edf"
    write_edf(p, sigs, n_records, data=data)
    return p


def test_night_quality_marks_the_flat_epoch(tmp_path):
    p = synthetic_night(tmp_path, n_epochs=3, flat_epoch=1)
    q = sq.night_quality(p)
    assert q.flat_epoch.tolist() == [False, True, False]
    assert q.loc[1, "HB_1_std_uv"] == 0.0 and q.loc[0, "HB_1_std_uv"] > 1
    assert q.loc[1, "HB_1_sat_share"] == 1.0  # -32767 digital is within 1 % of the range limit


def test_movement_per_epoch_needs_the_accelerometer_signals(tmp_path):
    assert sq.movement_per_epoch(synthetic_night(tmp_path)) is None
    sigs = [dict(label=a, samples=10, pmin=-2.0, pmax=2.0, dmin=-1000, dmax=1000, unit="g") for a in sq.ACCEL]
    rng = np.random.default_rng(1)
    data = [np.concatenate([rng.integers(-200, 200, 10) * (5 if r >= 30 else 1) for _ in sq.ACCEL]) for r in range(60)]
    p = tmp_path / "imu.edf"
    write_edf(p, sigs, 60, data=data)
    mv = sq.movement_per_epoch(p)
    assert len(mv) == 2 and mv[1] > mv[0]  # the second epoch moves more


def test_align_checks_the_number_of_epochs():
    night = pd.DataFrame({"hb_ai": [2, -2, 3, 0]})
    q = pd.DataFrame({"flat_epoch": [False, True, False, False]})
    out = sq.align(night, q)
    assert out.missing.tolist() == [False, True, False, False] and out.staged.tolist() == [True, False, True, True]
    assert len(sq.align(night, q.iloc[:3])) == 3  # one partial epoch of difference is accepted
    with pytest.raises(ValueError):
        sq.align(night, q.iloc[:2])


def joined_frame(missing, flat, std):
    missing = np.array(missing, dtype=bool)
    return pd.DataFrame({"missing": missing, "staged": ~missing, "flat_epoch": flat,
                         **{f"{ch}_{f}": 0.0 for ch in sq.CHANNELS for f in sq.FEATURES},
                         "HB_1_std_uv": std, "HB_1_flat_share": np.where(missing, 1.0, 0.0)})


def test_night_q1_overlap_and_medians():
    j = joined_frame([0, 1, 1, 0], [False, True, True, True], [10.0, 0.0, 0.0, 12.0])
    q = sq.night_q1(j)
    assert q["n_missing"] == 2 and q["n_flat"] == 3
    assert q["share_of_missing_that_are_flat"] == 1.0 and q["share_of_flat_that_are_missing"] == pytest.approx(2 / 3)
    assert q["median_when_missing"]["HB_1_std_uv"] == 0.0 and q["median_when_staged"]["HB_1_std_uv"] == 11.0
    assert sq.night_q1(joined_frame([0, 0], [False, False], [1.0, 1.0]))["share_of_missing_that_are_flat"] is None


def test_pooled_auroc_for_a_feature_that_separates_perfectly():
    j = joined_frame([0, 0, 1, 1], [False] * 4, [20.0, 30.0, 0.0, 0.0])
    out = sq.pooled_auroc([j, j])
    assert out["auroc_by_feature"]["HB_1_flat_share"] == 1.0 and out["auroc_by_feature"]["HB_1_std_uv"] == 0.0
    assert out["n_missing"] == 4


def test_near_far_staged_uses_distance_to_the_missing_epochs():
    missing = [0] * 10 + [1] + [0] * 40
    std = np.arange(51, dtype=float)
    j = joined_frame(missing, [False] * 51, std)
    out = sq.near_far_staged(j)
    assert out["n_near"] == 10 and out["n_far"] == 11  # near: epochs 5-9 and 11-15; far: epochs 40-50 (at least 30 away)
    assert out["median_near"]["HB_1_std_uv"] == pytest.approx(np.median(np.r_[std[5:10], std[11:16]]))


def test_movement_ratios_compare_the_five_epochs_before_a_long_run_with_quiet_epochs():
    missing = np.array([0] * 40 + [1] * 12 + [0] * 40, dtype=bool)
    movement = np.ones(len(missing))
    movement[35:40] = 4.0
    ratios = sq.movement_ratios(missing, ~missing, movement)
    assert ratios == [pytest.approx(4.0)]
    short = np.array([0] * 40 + [1] * 3 + [0] * 40, dtype=bool)
    assert sq.movement_ratios(short, ~short, np.ones(len(short))) == []  # shorter than a long run
    assert sq.movement_ratios(np.zeros(5, dtype=bool), np.ones(5, dtype=bool), np.ones(5)) == []  # no far epochs


def test_movement_ranks_compare_with_windows_of_the_same_night():
    missing = np.array([0] * 60 + [1] * 12 + [0] * 40, dtype=bool)
    quiet, loud = np.ones(len(missing)), np.ones(len(missing))
    loud[55:60] = 10.0
    assert sq.movement_ranks(missing, ~missing, quiet) == [1.0]  # equal to every window: nothing is quieter, all are equal
    assert sq.movement_ranks(missing, ~missing, loud) == [1.0]  # louder than all other windows
    low = np.ones(len(missing)) * 5.0
    low[55:60] = 0.1
    assert sq.movement_ranks(missing, ~missing, low) == [0.0]  # quieter than every comparable window
    assert sq.movement_ranks(np.zeros(5, dtype=bool), np.ones(5, dtype=bool), np.ones(5)) == []  # no long run


def test_movement_ranks_skip_runs_whose_preceding_epochs_are_not_staged():
    missing = np.array([0] * 40 + [1] * 12 + [0] * 40, dtype=bool)
    staged = ~missing
    staged[36] = False
    assert sq.movement_ranks(missing, staged, np.ones(len(missing))) == []


def test_order_check_compares_start_order_with_number_order():
    starts = {"sub-2": pd.Timestamp("2022-01-02"), "sub-10": pd.Timestamp("2022-01-01"), "sub-3": pd.Timestamp("2022-05-01"),
              "sub-4": pd.Timestamp("2022-06-01")}
    out = sq.order_check(starts, {"sub-2": "a", "sub-10": "a", "sub-3": "b", "sub-4": "b"})
    by = {o["pid"]: o for o in out}
    assert by["a"]["same_order"] is False and by["a"]["by_start"] == ["sub-10", "sub-2"]
    assert by["b"]["same_order"] is True


def test_run_on_a_tiny_dataset(tmp_path):
    # two nights in a tiny dataset: one complete (matches its key), one not downloaded
    data, edf_dir = tmp_path / "data", tmp_path / "edfs"
    p = synthetic_night(tmp_path, n_epochs=3, flat_epoch=1)
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    for sub in ("sub-1", "sub-2"):
        night = data / sub / "eeg"
        night.mkdir(parents=True)
        key = (len(p.read_bytes()), good)
        os.symlink(f"x/SHA256E-s{key[0]}--{key[1]}.edf", night / f"{sub}_task-Sleep_acq-headband_eeg.edf")
        psg = pd.DataFrame({"onset": [0, 30, 60], "stage_hum": [0, 2, 2], "stage_ai": [0, 2, 2]})
        hb = pd.DataFrame({"onset": [0, 30, 60], "stage_ai": [0, -2, 2]})
        psg.to_csv(night / f"{sub}_task-Sleep_acq-psg_events.tsv", sep="\t", index=False)
        hb.to_csv(night / f"{sub}_task-Sleep_acq-headband_events.tsv", sep="\t", index=False)
    (data / "participants.tsv").write_text("participant_id\tage\tsex\tbmi\tpid\nsub-1\t30\tF\t20\t1\nsub-2\t31\tF\t20\t1\n")
    target = sq.edf_path(edf_dir, "sub-1")
    target.parent.mkdir(parents=True)
    target.write_bytes(p.read_bytes())
    out = sq.run(data, edf_dir)
    assert out["nights_used"] == ["sub-1"] and out["n_nights_skipped_incomplete_or_missing"] == 1
    q = out["q1_per_night"]["sub-1"]
    assert q["share_of_missing_that_are_flat"] == 1.0 and q["share_of_flat_that_are_missing"] == 1.0
