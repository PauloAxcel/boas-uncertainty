"""Re-run every check behind docs/data_inventory.md sections 2-8.

Each check compares what the files contain with what the inventory documents
(DOCUMENTED below). Writes results/inventory_audit.json and exits non-zero if
any check fails, so a doc/data mismatch cannot go unnoticed.

Run from the repo root:  uv run python -m boas_unc.audit_inventory
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from boas_unc.inventory_figures import DATA
from boas_unc.load import load_all, runs, subjects

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "inventory_audit.json"

EPOCH_S, FS = 30, 256

# Values as written in docs/data_inventory.md. Change them only together with the doc.
DOCUMENTED = {
    "n_recordings": 128,
    "files_per_recording": sorted([
        "{s}_scans.tsv",
        *(f"eeg/{{s}}_task-Sleep_acq-{acq}_{suffix}" for acq in ("psg", "headband")
          for suffix in ("eeg.edf", "eeg.json", "channels.tsv", "events.tsv", "events.json")),
    ]),
    "events_columns": {
        "psg": ["onset", "duration", "begsample", "endsample", "offset", "stage_hum", "stage_ai"],
        "headband": ["onset", "duration", "begsample", "endsample", "offset", "stage_ai"],
    },
    "channels_columns": ["name", "type", "units", "sampling_frequency"],
    "eeg_json_constants": {
        "psg": {"Manufacturer": "Micromed", "ManufacturersModelName": "SD plus", "SamplingFrequency": 256,
                "PowerLineFrequency": 50, "EEGReference": "M1 (left mastoid)", "SoftwareFilters": "n/a"},
        "headband": {"Manufacturer": "Bitbrain Technologies", "ManufacturersModelName": "IKON Sleep",
                     "SamplingFrequency": 256, "PowerLineFrequency": 50, "EEGChannelCount": 2,
                     "EEGReference": "M1 (left mastoid)", "SoftwareFilters": "n/a"},
    },
    "psg_eeg_channel_count": {"8": 64, "9": 2, "11": 38, "12": 5, "13": 19},
    "channel_name_counts": {
        **{f"PSG_{c}": 128 for c in ("F3", "F4", "C3", "C4", "O1", "O2", "EMG")},
        "PSG_EOG": 102, "PSG_EOGL": 26, "PSG_EOGR": 26,
        "PSG_THER": 62, "PSG_THOR": 62, "PSG_ABD": 62, "PSG_CAN": 19,
        "PSG_PULSE": 104, "PSG_BEAT": 104, "PSG_SPO2": 104,
        "HB_1": 128, "HB_2": 128, **{f"HB_IMU_{i}": 94 for i in range(1, 7)}, "HB_PULSE": 112,
    },
    "label_counts": {
        "psg_hum": {"0": 19137, "1": 4462, "2": 72181, "3": 5225, "4": 18754, "8": 336},
        "psg_ai": {"-2": 550, "0": 19543, "1": 3137, "2": 71621, "3": 5178, "4": 20066},
        "hb_ai": {"-2": 4881, "0": 18919, "1": 2202, "2": 70543, "3": 4068, "4": 19482},
    },
    "allowed_codes": {"psg_hum": [0, 1, 2, 3, 4, 8], "psg_ai": [-2, 0, 1, 2, 3, 4], "hb_ai": [-2, 0, 1, 2, 3, 4]},
    "nights_with_code": {
        "psg_hum": {"0": 128, "1": 128, "2": 128, "3": 91, "4": 128, "8": 37},
        "psg_ai": {"-2": 50, "0": 128, "1": 128, "2": 128, "3": 78, "4": 128},
        "hb_ai": {"-2": 100, "0": 128, "1": 128, "2": 128, "3": 72, "4": 127},
    },
    "special_runs": {  # n runs, position counts, min / median / max run length
        "psg_hum=8": {"n": 50, "where": {"middle": 48, "end": 2}, "len": [1, 6, 46]},
        "psg_ai=-2": {"n": 115, "where": {"middle": 110, "end": 5}, "len": [1, 2, 85]},
        "hb_ai=-2": {"n": 372, "where": {"middle": 302, "end": 64, "start": 6}, "len": [1, 1, 670]},
    },
    "epochs_total": 120095,
    "epochs_per_night_min_max": [409, 1119],
    "duration_minus_epochs_range": [0, 29],
    "participants": {"rows": 128, "unique_pid": 100, "F": 76, "M": 52,
                     "na": {"sub-16": ["age"], "sub-29": ["bmi"]}, "multi_night_pids": 20},
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, section: str, name: str, actual, expected=True):
        self.checks.append({"section": section, "check": name, "passed": actual == expected,
                            "actual": actual, "expected": expected})


def run_lengths(seq, code) -> list[tuple[int, int]]:
    return runs(seq, code)


def audit(data: Path) -> Audit:
    a = Audit()
    subs = subjects(data)

    # Section 2: folder structure
    a.check("2", "number of sub-* folders", len(subs), DOCUMENTED["n_recordings"])
    layouts = Counter(
        tuple(sorted(str(p.relative_to(data / s)).replace(s, "{s}") for p in (data / s).rglob("*") if not p.is_dir()))
        for s in subs)
    a.check("2", "every folder has the same 11 files", [list(k) for k in layouts],
            [DOCUMENTED["files_per_recording"]])
    edfs = list(data.glob("sub-*/eeg/*.edf"))
    a.check("intro", "EDFs are annex symlinks without content (not downloaded)",
            sum(p.is_symlink() and not p.exists() for p in edfs), len(edfs))

    # Section 3: events files
    for acq in ("psg", "headband"):
        headers, bad_timing, durations, offsets = Counter(), [], Counter(), Counter()
        for s in subs:
            ev = read_tsv(data / s / "eeg" / f"{s}_task-Sleep_acq-{acq}_events.tsv")
            headers[tuple(ev.columns)] += 1
            durations.update(ev["duration"].tolist())
            offsets.update(ev["offset"].tolist())
            i = pd.RangeIndex(len(ev))
            ok = ((ev["onset"] == i * EPOCH_S).all()
                  and (ev["begsample"] == i * EPOCH_S * FS + 1).all()
                  and (ev["endsample"] == (i + 1) * EPOCH_S * FS).all())
            if not ok:
                bad_timing.append(s)
            sidecar = read_json(data / s / "eeg" / f"{s}_task-Sleep_acq-{acq}_events.json")
            if set(sidecar) != set(ev.columns):
                bad_timing.append(f"{s}: sidecar keys != columns")
        a.check("3", f"{acq} events columns", [list(h) for h in headers], [DOCUMENTED["events_columns"][acq]])
        a.check("3", f"{acq} duration always 30", sorted(durations), [EPOCH_S])
        a.check("3", f"{acq} offset always 0", sorted(offsets), [0])
        a.check("3", f"{acq} onset=30i, samples 1-based contiguous, sidecar keys match", bad_timing, [])

    # Section 4: sidecars and channel files
    for acq in ("psg", "headband"):
        consts, keysets = DOCUMENTED["eeg_json_constants"][acq], Counter()
        wrong = []
        for s in subs:
            j = read_json(data / s / "eeg" / f"{s}_task-Sleep_acq-{acq}_eeg.json")
            keysets[tuple(sorted(j))] += 1
            wrong += [f"{s}.{k}={j.get(k)!r}" for k, v in consts.items() if j.get(k) != v]
        a.check("4", f"{acq}_eeg.json has one key set", len(keysets), 1)
        a.check("4", f"{acq}_eeg.json documented constant values", wrong, [])
    counts = Counter(str(read_json(data / s / "eeg" / f"{s}_task-Sleep_acq-psg_eeg.json")["EEGChannelCount"])
                     for s in subs)
    a.check("4", "psg EEGChannelCount distribution", dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
            DOCUMENTED["psg_eeg_channel_count"])

    names, headers, fs, count_vs_eeg_rows = Counter(), Counter(), Counter(), []
    for s in subs:
        for acq in ("psg", "headband"):
            ch = read_tsv(data / s / "eeg" / f"{s}_task-Sleep_acq-{acq}_channels.tsv")
            headers[tuple(ch.columns)] += 1
            names.update(ch["name"])
            fs.update(ch["sampling_frequency"])
            j = read_json(data / s / "eeg" / f"{s}_task-Sleep_acq-{acq}_eeg.json")
            if j["EEGChannelCount"] != (ch["type"] == "EEG").sum():
                count_vs_eeg_rows.append(f"{s}/{acq}")
    a.check("4", "channels.tsv columns", [list(h) for h in headers], [DOCUMENTED["channels_columns"]])
    a.check("4", "all channels sampled at 256 Hz", sorted(fs), [FS])
    a.check("4", "channel names and counts (exact, with prefixes)", dict(sorted(names.items())),
            dict(sorted(DOCUMENTED["channel_name_counts"].items())))
    a.check("4", "EEGChannelCount equals number of type=EEG rows", count_vs_eeg_rows, [])

    groups = {"PSG_EEG": ["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"],
              "PSG_EOG": ["PSG_EOG", "PSG_EOGL", "PSG_EOGR"], "PSG_EMG": ["PSG_EMG"],
              "PSG_BELTS": ["PSG_ABD", "PSG_THOR"], "PSG_THER": ["PSG_THER"], "PSG_CAN": ["PSG_CAN"],
              "PSG_PPG": ["PSG_PULSE", "PSG_BEAT", "PSG_SPO2"], "HB_EEG": ["HB_1", "HB_2"],
              "HB_IMU": [f"HB_IMU_{i}" for i in range(1, 7)], "HB_PULSE": ["HB_PULSE"]}
    top = read_tsv(data / "channels.tsv").set_index("REC")
    mismatch = []
    for s in subs:
        present = set()
        for acq in ("psg", "headband"):
            present |= set(read_tsv(data / s / "eeg" / f"{s}_task-Sleep_acq-{acq}_channels.tsv")["name"])
        mismatch += [f"{s}.{g}" for g, chs in groups.items() if int(top.loc[s, g]) != int(bool(present & set(chs)))]
    a.check("4", "top-level channels.tsv flags match per-recording channels", mismatch, [])
    scans = Counter(tuple(f.split("acq-")[1] for f in read_tsv(data / s / f"{s}_scans.tsv")["filename"])
                    for s in subs)
    a.check("4", "scans.tsv lists psg then headband EDF", [list(k) for k in scans],
            [["psg_eeg.edf", "headband_eeg.edf"]])

    # Section 5: label values
    df = load_all(data)
    a.check("5", "total epochs per label column", len(df), DOCUMENTED["epochs_total"])
    for col in ("psg_hum", "psg_ai", "hb_ai"):
        vc = df[col].value_counts()
        a.check("5", f"{col} counts", {str(k): int(v) for k, v in sorted(vc.items())},
                DOCUMENTED["label_counts"][col])
        a.check("5", f"{col} only uses its documented codes", sorted(int(c) for c in vc.index),
                DOCUMENTED["allowed_codes"][col])
        nights = df.groupby(df[col])["sub"].nunique()
        a.check("5", f"{col} nights containing each code", {str(k): int(v) for k, v in sorted(nights.items())},
                DOCUMENTED["nights_with_code"][col])
    for col, code in (("psg_hum", 8), ("psg_ai", -2), ("hb_ai", -2)):
        where, lengths = Counter(), []
        for _, night in df.groupby("sub"):
            seq = night.sort_values("onset")[col].to_numpy()
            for start, length in run_lengths(seq, code):
                where["start" if start == 0 else "end" if start + length == len(seq) else "middle"] += 1
                lengths.append(length)
        lengths.sort()
        a.check("5", f"{col}={code} runs", {"n": len(lengths), "where": dict(where),
                                            "len": [lengths[0], lengths[len(lengths) // 2], lengths[-1]]},
                DOCUMENTED["special_runs"][f"{col}={code}"])

    # Sections 6-7: timing and pairing
    per_night = df.groupby("sub").size()
    a.check("6", "epochs per night min/max", [int(per_night.min()), int(per_night.max())],
            DOCUMENTED["epochs_per_night_min_max"])
    diffs, pairing = [], []
    for s in subs:
        pj = read_json(data / s / "eeg" / f"{s}_task-Sleep_acq-psg_eeg.json")
        hj = read_json(data / s / "eeg" / f"{s}_task-Sleep_acq-headband_eeg.json")
        pe = read_tsv(data / s / "eeg" / f"{s}_task-Sleep_acq-psg_events.tsv")
        he = read_tsv(data / s / "eeg" / f"{s}_task-Sleep_acq-headband_events.tsv")
        diffs.append(pj["RecordingDuration"] - EPOCH_S * len(pe))
        if not (pe["onset"].equals(he["onset"]) and pj["RecordingDuration"] == hj["RecordingDuration"]):
            pairing.append(s)
    a.check("6", "RecordingDuration - 30*n_epochs range", [min(diffs), max(diffs)],
            DOCUMENTED["duration_minus_epochs_range"])
    a.check("7", "psg/headband: same onsets and RecordingDuration", pairing, [])

    # Section 8: participants
    pt = pd.read_csv(data / "participants.tsv", sep="\t", dtype=str, keep_default_na=False)  # keep literal "n/a"
    doc = DOCUMENTED["participants"]
    a.check("8", "participants rows", len(pt), doc["rows"])
    a.check("8", "participant_id set equals sub-* folders", sorted(pt["participant_id"]) == sorted(subs))
    a.check("8", "unique pid", pt["pid"].nunique(), doc["unique_pid"])
    a.check("8", "sex counts", {k: int(v) for k, v in pt["sex"].value_counts().sort_index().items()},
            {"F": doc["F"], "M": doc["M"]})
    na = {r.participant_id: [c for c in ("age", "sex", "bmi", "pid") if getattr(r, c) == "n/a"]
          for r in pt.itertuples() if "n/a" in (r.age, r.sex, r.bmi, r.pid)}
    a.check("8", "n/a values", na, doc["na"])
    a.check("8", "pids with more than one night", int((pt["pid"].value_counts() > 1).sum()),
            doc["multi_night_pids"])
    return a


def main():
    a = audit(DATA)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(a.checks, indent=2, default=str))
    failed = [c for c in a.checks if not c["passed"]]
    for c in a.checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] §{c['section']} {c['check']}")
    print(f"{len(a.checks) - len(failed)}/{len(a.checks)} checks passed; wrote {OUT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
