# boas-uncertainty

Where does a 2-channel sleep headband disagree with clinical polysomnography (PSG), and can those epochs be flagged using what the device itself produces?

This is an independent analysis of the open **Bitbrain Open Access Sleep (BOAS)** dataset. It is not affiliated with, or endorsed by, Bitbrain. The code and the derived results are mine; the dataset belongs to its authors (see "Data").

**AI assistance.** Developed with AI assistance: Claude Code helped write the code and documents, and a second AI model (named "Argon" in `CHECKLIST.md`) reviewed the work. Every result is computed by the code in this repository from the dataset, and the tests check the code. Decisions and their reviews are recorded in `CHECKLIST.md`.

**Start with [`REPORT.md`](REPORT.md).** It is generated from the files in `results/` and holds the findings, the figures and the limitations. `docs/definitions.md` defines every metric before it was implemented, and `CHECKLIST.md` records the decisions taken and the questions that only the dataset's authors can answer.

## What it does

1. **Agreement** of the headband AI labels with the human consensus, five-stage and three-class (wake, light, deep), counting epochs with no stage in two ways.
2. **Where errors concentrate:** by stage, near stage changes, near artefact flags, by hour of the night and by person.
3. **Clinical impact:** how the disagreements change total sleep time, sleep efficiency, latencies, WASO and stage shares, night by night (Bland-Altman).
4. **Device-only abstention:** flags built from the headband's own label sequence, with agreement against coverage curves, and a check of whether a learned model on the labels does better.
5. **Raw signal:** what the headband signal looks like when the headband gives no stage (flat channels, one clean channel, movement), read from the EDF files.

Per-night medians and IQRs are reported next to pooled figures. Every number in `REPORT.md` is read from `results/*.json` by `src/boas_unc/report.py`, and a test checks that no result is typed by hand.

## Data

The data is not in this repository and is not redistributed.

- Dataset: OpenNeuro `ds005555`, `doi:10.18112/openneuro.ds005555.v1.1.3`, licence CC0. Authors: López-Larraz, Sierra-Torralba, Clemente, Fierro, Oriol, Minguez, Montesano, Klinzing. Cite the dataset and its authors if you use it.
- Labels (small, Phase 0):

  ```bash
  git clone --depth 1 https://github.com/OpenNeuroDatasets/ds005555.git data/boas
  ```

  The code reads the labels from `data/boas/` and never writes there.
- Headband EDFs (about 13 GB, needed only for the raw-signal analyses): fetch them with `git annex` or DataLad from the same dataset into a folder of your choice. The code checks each file's size and SHA256 against the git-annex key before using it, and skips incomplete files. Point it at the folder:

  ```bash
  export BOAS_EDF_DIR=/path/to/headband_edfs   # default: ./headband_edfs
  ```

  Expected layout: `<BOAS_EDF_DIR>/sub-N/eeg/sub-N_task-Sleep_acq-headband_eeg.edf`.

## Run

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest

# label-only analyses
uv run python -m boas_unc.agreement
uv run python -m boas_unc.hotspots
uv run python -m boas_unc.clinical
uv run python -m boas_unc.abstain
uv run python -m boas_unc.learned_abstain
uv run python -m boas_unc.person_effects
uv run python -m boas_unc.three_class

# raw-signal analyses (need the EDF files)
uv run python -m boas_unc.signal_quality
uv run python -m boas_unc.one_channel
uv run python -m boas_unc.stage_signature

uv run python -m boas_unc.report      # rewrites REPORT.md from results/
```

Some modules also write figures into `figures/`. `uv run python -m boas_unc.plot_night --help` draws a full-night overview for one night.

## Layout

```
src/boas_unc/   load, agreement, hotspots, metrics, clinical, abstain, learned_abstain,
                person_effects, three_class, edf (numpy EDF reader), signal_quality,
                one_channel, stage_signature, plot_night, report, figure scripts
tests/          one test module per source module
docs/           data_inventory.md (what the files contain), definitions.md (metric definitions)
results/        JSON outputs (committed)
figures/        figures used by REPORT.md
```

## Project rules

The analysis followed these rules, which the code and tests enforce where possible:

- Python 3.11 with uv; libraries limited to pandas, numpy, scipy, scikit-learn, matplotlib and pytest.
- The dataset is read-only. Column names, stage codes and file names are read from the files and the dataset's JSON sidecars, never assumed, and documented in `docs/data_inventory.md`.
- Every metric is defined in `docs/definitions.md` before it is implemented.
- Every number in `REPORT.md` comes from a file in `results/`; none is typed by hand.
- Per-night distributions (median, IQR) are reported next to pooled figures.
- Functions are small and pure, and each module has tests.

## Limitations

One dataset, 128 nights from 100 healthy adults, recorded in one lab setting. The human consensus is itself a judgement, and which model produced the shipped headband labels, and how, is not stated in the dataset (`CHECKLIST.md`, C2 and C7). Findings about why a channel fails are consistent with the signal but were not verified. See the limitations section of `REPORT.md`.

## Licence

Code: MIT (see `LICENSE`). Derived results and figures: derived from CC0 data; please cite the dataset authors.
