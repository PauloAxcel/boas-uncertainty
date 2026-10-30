import numpy as np
import pandas as pd
import pytest

from boas_unc import load


def events(onsets, hum=None, ai=None):
    """A small events table; `hum` is only given for the PSG file."""
    df = pd.DataFrame({"onset": onsets, "duration": 30, "stage_ai": ai})
    if hum is not None:
        df["stage_hum"] = hum
    return df


def write_night(data, sub, psg, hb):
    eeg = data / sub / "eeg"
    eeg.mkdir(parents=True)
    psg.to_csv(eeg / f"{sub}_task-Sleep_acq-psg_events.tsv", sep="\t", index=False)
    hb.to_csv(eeg / f"{sub}_task-Sleep_acq-headband_events.tsv", sep="\t", index=False)


# A 5-epoch night with a disconnection (8) and an AI artefact (-2).
PSG = events([0, 30, 60, 90, 120], hum=[0, 1, 8, 2, 4], ai=[0, 1, -2, 2, 4])
HB = events([0, 30, 60, 90, 120], ai=[0, -2, 0, 2, 4])


def test_subjects_sorted_by_number(tmp_path):
    for name in ["sub-10", "sub-2", "sub-1"]:
        (tmp_path / name).mkdir()
    (tmp_path / "sub-3").write_text("a file, not a folder")
    (tmp_path / "other").mkdir()
    assert load.subjects(tmp_path) == ["sub-1", "sub-2", "sub-10"]


def test_runs_start_and_length():
    seq = np.array([-2, -2, 1, 1, -2, 3, -2, -2, -2])
    assert load.runs(seq, -2) == [(0, 2), (4, 1), (6, 3)]
    assert load.runs(seq, 4) == []


def test_align_sources_aligned_night():
    night = load.align_sources(PSG, HB)
    assert list(night.columns) == ["onset", "psg_hum", "psg_ai", "hb_ai"]
    assert night.psg_hum.tolist() == [0, 1, 8, 2, 4]
    assert night.hb_ai.tolist() == [0, -2, 0, 2, 4]


def test_align_sources_uses_onset_not_row_order():
    shuffled = HB.iloc[::-1].reset_index(drop=True)
    night = load.align_sources(PSG, shuffled)
    assert night.onset.tolist() == [0, 30, 60, 90, 120]
    assert night.hb_ai.tolist() == [0, -2, 0, 2, 4]


def test_align_sources_misaligned_keeps_one_sided_rows():
    hb = events([0, 30, 90, 120, 150], ai=[0, -2, 2, 4, 3])  # no 60, extra 150
    night = load.align_sources(PSG, hb)
    assert night.onset.tolist() == [0, 30, 60, 90, 120, 150]
    assert np.isnan(night.hb_ai[night.onset == 60].item())
    assert np.isnan(night.psg_hum[night.onset == 150].item())


def test_align_sources_rejects_repeated_onset():
    hb = events([0, 30, 30, 90, 120], ai=[0, -2, 0, 2, 4])
    with pytest.raises(pd.errors.MergeError):
        load.align_sources(PSG, hb)


def test_stage_labels_maps_stages_and_blanks_special_codes():
    codes = pd.Series([0, 1, 2, 3, 4, 8, -2, np.nan])
    labels = load.stage_labels(codes)
    assert labels.iloc[:5].tolist() == ["W", "N1", "N2", "N3", "REM"]
    assert labels.iloc[5:].isna().all()


def test_flag_special_marks_artefact_and_disconnection():
    flagged = load.flag_special(load.align_sources(PSG, HB))
    assert flagged.hum_disconnected.tolist() == [False, False, True, False, False]
    assert flagged.psg_ai_missing.tolist() == [False, False, True, False, False]
    assert flagged.hb_ai_missing.tolist() == [False, True, False, False, False]
    assert flagged.aligned.all()


def test_flag_special_does_not_drop_or_change_rows():
    night = load.align_sources(PSG, HB)
    flagged = load.flag_special(night)
    assert len(flagged) == len(night)
    assert flagged[load.SOURCE_COLUMNS].equals(night[load.SOURCE_COLUMNS])


def test_flag_special_aligned_is_false_on_one_sided_rows():
    hb = events([0, 30, 90, 120], ai=[0, -2, 2, 4])
    flagged = load.flag_special(load.align_sources(PSG, hb))
    assert flagged.aligned.tolist() == [True, True, False, True, True]


def test_add_stage_labels_columns():
    out = load.add_stage_labels(load.align_sources(PSG, HB))
    assert out.hum_stage.tolist()[:2] == ["W", "N1"]
    assert pd.isna(out.hum_stage[2])  # code 8
    assert pd.isna(out.hb_ai_stage[1])  # code -2
    assert out.psg_ai_stage.tolist()[3:] == ["N2", "REM"]


def test_read_events_and_load_night(tmp_path):
    write_night(tmp_path, "sub-1", PSG, HB)
    psg, hb = load.read_events(tmp_path, "sub-1")
    assert len(psg) == len(hb) == 5
    night = load.load_night(tmp_path, "sub-1")
    assert night["sub"].eq("sub-1").all()
    assert night.epoch.tolist() == [0, 1, 2, 3, 4]
    assert night.onset.tolist() == [0, 30, 60, 90, 120]
    assert night.hum_disconnected.sum() == 1 and night.hb_ai_missing.sum() == 1
    assert len(night) == len(PSG)  # nothing dropped


def test_read_participants_keeps_strings(tmp_path):
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\tbmi\tpid\nsub-1\tn/a\tF\t19.8\t89\n")
    pt = load.read_participants(tmp_path)
    assert pt.age.tolist() == ["n/a"] and pt.pid.tolist() == ["89"]


def test_attach_pid_maps_nights_to_people():
    table = pd.DataFrame({"sub": ["sub-1", "sub-1", "sub-2"], "epoch": [0, 1, 0]})
    pt = pd.DataFrame({"participant_id": ["sub-1", "sub-2"], "pid": ["89", "89"]})
    out = load.attach_pid(table, pt)
    assert out.pid.tolist() == ["89", "89", "89"]
    assert list(out.columns)[:2] == ["sub", "pid"]


def test_load_all_stacks_nights_with_pid(tmp_path):
    write_night(tmp_path, "sub-2", PSG, HB)
    write_night(tmp_path, "sub-1", PSG, HB)
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\tbmi\tpid\nsub-1\t27\tF\t19.8\t89\nsub-2\t27\tF\t19.8\t89\n")
    table = load.load_all(tmp_path)
    assert table["sub"].unique().tolist() == ["sub-1", "sub-2"]
    assert len(table) == 10
    assert table.pid.eq("89").all()


def test_alignment_report_flags_misaligned_night():
    ok = load.add_stage_labels(load.flag_special(load.align_sources(PSG, HB)))
    ok.insert(0, "sub", "sub-1")
    hb = events([0, 30, 90, 120, 150], ai=[0, -2, 2, 4, 3])
    bad = load.add_stage_labels(load.flag_special(load.align_sources(PSG, hb)))
    bad.insert(0, "sub", "sub-2")
    report = load.alignment_report(pd.concat([ok, bad], ignore_index=True))
    assert report.loc["sub-1"].tolist() == [5, 0, 0, True]
    assert report.loc["sub-2"].tolist() == [6, 1, 1, False]
