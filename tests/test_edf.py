import numpy as np
import pytest

from boas_unc import edf


def field(text, width):
    return str(text).ljust(width)[:width].encode("ascii")


def write_edf(path, signals, n_records, record_s=1.0, start=("14.11.22", "23.57.16"), data=None):
    """A tiny EDF file. signals: list of dicts with label, samples (per record), physical/digital ranges."""
    ns = len(signals)
    fixed = (field("0", 8) + field("sub-x", 80) + field("sub-x", 80) + field(start[0], 8) + field(start[1], 8)
             + field(256 + 256 * ns, 8) + field("", 44) + field(n_records, 8) + field(record_s, 8) + field(ns, 4))
    per = b""
    for name, width in edf.SIGNAL_FIELDS:
        for s in signals:
            value = {"label": s["label"], "transducer": "", "physical_dimension": s.get("unit", "uV"),
                     "physical_min": s["pmin"], "physical_max": s["pmax"], "digital_min": s["dmin"],
                     "digital_max": s["dmax"], "prefilter": "", "samples_per_record": s["samples"],
                     "reserved": ""}[name]
            per += field(value, width)
    body = b""
    if data is not None:
        body = np.asarray(data, dtype="<i2").tobytes()
    path.write_bytes(fixed + per + body)


SIGS = [dict(label="HB_1", samples=4, pmin=-100.0, pmax=100.0, dmin=-32768, dmax=32767),
        dict(label="HB_IMU_1", samples=2, pmin=-2.0, pmax=2.0, dmin=-1000, dmax=1000, unit="g")]


def two_record_data():
    # record 0: HB_1 = 0,1,2,3 then IMU = 10,20; record 1: HB_1 = 4,5,6,7 then IMU = 30,40
    return [[0, 1, 2, 3, 10, 20], [4, 5, 6, 7, 30, 40]]


def test_parse_start_uses_the_edf_century_rule():
    assert edf.parse_start("14.11.22", "23.57.16").year == 2022
    assert edf.parse_start("01.01.99", "00.00.00").year == 1999


def test_read_header_fields(tmp_path):
    p = tmp_path / "a.edf"
    write_edf(p, SIGS, 2, data=two_record_data())
    h = edf.read_header(p)
    assert h["n_signals"] == 2 and h["n_records"] == 2 and h["record_s"] == 1.0
    assert h["start"].isoformat() == "2022-11-14T23:57:16" and h["patient"] == "sub-x"
    assert [s["label"] for s in h["signals"]] == ["HB_1", "HB_IMU_1"]
    assert h["signals"][0]["sampling_hz"] == 4.0 and h["signals"][1]["sampling_hz"] == 2.0
    assert h["signals"][1]["physical_dimension"] == "g" and edf.duration_s(h) == 2.0


def test_parse_header_rejects_an_inconsistent_header_size():
    raw = bytearray(512)
    raw[184:192] = b"999     "
    raw[252:256] = b"1   "
    raw[0:8] = b"0       "
    raw[168:184] = b"14.11.2223.57.16"
    with pytest.raises(ValueError):
        edf.parse_header(bytes(raw))


def test_read_digital_picks_the_right_samples_for_each_signal(tmp_path):
    p = tmp_path / "a.edf"
    write_edf(p, SIGS, 2, data=two_record_data())
    h = edf.read_header(p)
    assert edf.read_digital(p, h, "HB_1").tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert edf.read_digital(p, h, "HB_IMU_1").tolist() == [10, 20, 30, 40]


def test_physical_scaling_uses_the_digital_and_physical_ranges(tmp_path):
    p = tmp_path / "a.edf"
    write_edf(p, SIGS, 2, data=two_record_data())
    imu = edf.read_signal(p, "HB_IMU_1")
    assert imu == pytest.approx([0.02, 0.04, 0.06, 0.08])  # (d - -1000) * 4 / 2000 + -2
    sig = dict(physical_min=-100.0, physical_max=100.0, digital_min=-32768, digital_max=32767)
    assert edf.to_physical(np.array([-32768, 32767]), sig) == pytest.approx([-100.0, 100.0])


def test_unknown_label_is_an_error(tmp_path):
    p = tmp_path / "a.edf"
    write_edf(p, SIGS, 2, data=two_record_data())
    with pytest.raises(KeyError):
        edf.read_signal(p, "NOPE")


def test_a_truncated_file_is_refused(tmp_path):
    p = tmp_path / "a.edf"
    write_edf(p, SIGS, 3, data=two_record_data())  # header says 3 records, 2 are present
    with pytest.raises(ValueError):
        edf.read_signal(p, "HB_1")


def test_reading_never_modifies_the_file(tmp_path):
    p = tmp_path / "a.edf"
    write_edf(p, SIGS, 2, data=two_record_data())
    before = p.read_bytes()
    edf.read_signal(p, "HB_1")
    assert p.read_bytes() == before


def test_epochs_cuts_whole_epochs_and_drops_the_remainder():
    out = edf.epochs(np.arange(25.0), sampling_hz=1.0, epoch_s=10.0)
    assert out.shape == (2, 10) and out[1, 0] == 10.0
