"""A small read-only EDF reader (numpy only), for the headband files of Phase 3.

EDF layout: a 256-byte fixed header, then 256 bytes per signal (labels, transducer, physical
dimension, physical and digital ranges, prefilter, samples per record, reserved), then the data
records. Each record holds, for every signal in turn, its samples for one record duration as
little-endian 16-bit integers. Values are scaled to physical units from the digital and physical
ranges. Files are only ever opened for reading, and never copied.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

FIXED_HEADER = 256
PER_SIGNAL = 256
ANNOTATIONS = "EDF Annotations"
# (name, width in bytes) of the per-signal header fields, in file order.
SIGNAL_FIELDS = [("label", 16), ("transducer", 80), ("physical_dimension", 8), ("physical_min", 8),
                 ("physical_max", 8), ("digital_min", 8), ("digital_max", 8), ("prefilter", 80),
                 ("samples_per_record", 8), ("reserved", 32)]


def ascii_field(raw: bytes, start: int, width: int) -> str:
    return raw[start:start + width].decode("ascii", "replace").strip()


def parse_start(date: str, time: str) -> datetime:
    """EDF start date dd.mm.yy and time hh.mm.ss. Years from 85 on are 19xx, the rest 20xx (EDF rule)."""
    day, month, year = (int(x) for x in date.split("."))
    hour, minute, second = (int(x) for x in time.split("."))
    return datetime(1900 + year if year >= 85 else 2000 + year, month, day, hour, minute, second)


def parse_header(raw: bytes) -> dict:
    """Header fields and per-signal descriptions from the first bytes of an EDF file."""
    n_signals = int(ascii_field(raw, 252, 4))
    header_bytes = int(ascii_field(raw, 184, 8))
    if header_bytes != FIXED_HEADER + PER_SIGNAL * n_signals:
        raise ValueError(f"header size {header_bytes} does not fit {n_signals} signals")
    fields = {}
    offset = FIXED_HEADER
    for name, width in SIGNAL_FIELDS:
        fields[name] = [ascii_field(raw, offset + i * width, width) for i in range(n_signals)]
        offset += width * n_signals
    signals = []
    for i in range(n_signals):
        sig = {name: fields[name][i] for name, _ in SIGNAL_FIELDS}
        for name in ("physical_min", "physical_max", "digital_min", "digital_max"):
            sig[name] = float(sig[name])
        sig["samples_per_record"] = int(sig["samples_per_record"])
        signals.append(sig)
    record_s = float(ascii_field(raw, 244, 8))
    for sig in signals:
        sig["sampling_hz"] = sig["samples_per_record"] / record_s
    return {"version": ascii_field(raw, 0, 8), "patient": ascii_field(raw, 8, 80),
            "recording": ascii_field(raw, 88, 80),
            "start": parse_start(ascii_field(raw, 168, 8), ascii_field(raw, 176, 8)),
            "header_bytes": header_bytes, "n_records": int(ascii_field(raw, 236, 8)), "record_s": record_s,
            "n_signals": n_signals, "signals": signals}


def read_header(path: Path) -> dict:
    """Parse the header of an EDF file, reading only its first bytes."""
    with open(path, "rb") as f:
        head = f.read(FIXED_HEADER)
        n_signals = int(ascii_field(head, 252, 4))
        head += f.read(PER_SIGNAL * n_signals)
    return parse_header(head)


def duration_s(header: dict) -> float:
    return header["n_records"] * header["record_s"]


def signal_index(header: dict, label: str) -> int:
    labels = [s["label"] for s in header["signals"]]
    if label not in labels:
        raise KeyError(f"no signal {label!r}; the file has {labels}")
    return labels.index(label)


def to_physical(digital: np.ndarray, sig: dict) -> np.ndarray:
    """Digital values to physical units using the signal's digital and physical ranges."""
    gain = (sig["physical_max"] - sig["physical_min"]) / (sig["digital_max"] - sig["digital_min"])
    return (digital.astype(np.float64) - sig["digital_min"]) * gain + sig["physical_min"]


def read_digital(path: Path, header: dict, label: str) -> np.ndarray:
    """The raw 16-bit integers of one signal, for the whole recording."""
    per_record = [s["samples_per_record"] for s in header["signals"]]
    record_len = sum(per_record)
    expected = header["header_bytes"] + 2 * record_len * header["n_records"]
    if Path(path).stat().st_size < expected:
        raise ValueError(f"{path} is shorter than its header says ({Path(path).stat().st_size} < {expected} bytes)")
    data = np.memmap(path, dtype="<i2", mode="r", offset=header["header_bytes"],
                     shape=(header["n_records"], record_len))
    i = signal_index(header, label)
    start = sum(per_record[:i])
    return np.array(data[:, start:start + per_record[i]]).reshape(-1)


def read_signal(path: Path, label: str, header: dict | None = None) -> np.ndarray:
    """One signal of the whole recording in physical units (for example microvolts)."""
    header = header or read_header(path)
    return to_physical(read_digital(path, header, label), header["signals"][signal_index(header, label)])


def epochs(signal: np.ndarray, sampling_hz: float, epoch_s: float = 30.0) -> np.ndarray:
    """The signal cut into whole epochs: shape (n_epochs, samples per epoch). A partial last epoch is dropped."""
    size = int(round(sampling_hz * epoch_s))
    n = len(signal) // size
    return signal[:n * size].reshape(n, size)
