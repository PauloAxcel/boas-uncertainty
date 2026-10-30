# Definitions

Every clinical and agreement metric, defined before it is implemented (project rules). Decisions D1–D11 were locked on 2026-10-26; the reasoning and the data behind each one is in `CHECKLIST.md` under the same number. If a definition changes, change it here and in the checklist in the same edit.

Epoch length is 30 s. Stage codes (from `data/boas/README`, see `docs/data_inventory.md` section 5): 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM, `8` = PSG disconnected (human labels only), `-2` = artefact / missing (AI labels only).

## Conventions

- **Sleep epoch:** an epoch with code 1, 2, 3 or 4.
- **Unscored epoch:** headband `-2`. Neither sleep nor wake.
- **Excluded epoch:** human `8`. Outside time in bed, for both sequences (D2).
- `load.py` keeps every epoch on the full timeline and adds boolean flags. Nothing is dropped there.
- `metrics.py` functions take **one label sequence plus a boolean mask of valid epochs** and never look for code `8` themselves. Unscored (`-2`) epochs stay in the sequence and count in time in bed.
- All times are in minutes. Latencies are timestamp differences on the full timeline: (`onset` of target epoch − `onset` of reference epoch) / 60. Never count rows after filtering (D8).

## Two analyses (D3)

| | (i) Standalone, primary | (ii) Matched-epoch |
|---|---|---|
| Epochs | Each sequence on its full timeline | Only epochs where both sequences have a stage 0–4 |
| Time in bed | `TIB` | `TIB_matched` |
| Metrics | All | Stage-based metrics only: TST, sleep efficiency, stage percentages. No latencies, WASO or terminal wake |
| Nights | All 128, high-dropout nights flagged | Excludes nights with more than 20 % headband `-2`. Sensitivity: 10 % threshold, and all nights |

Decide the threshold before looking at clinical results.

## Sleep metrics (D2, D4, D5, D6, D8, D9, D11)

| Metric | Definition |
|---|---|
| **Time in bed (TIB)** | (recording length − human `8` epochs) in minutes. Recording start to end, because there are no lights markers (checklist A6). Headband `-2` stays inside. Same value for both sequences in (i) |
| **TIB_matched** | Number of epochs with a stage 0–4 in both sequences × 0.5 min. Used only in (ii) |
| **Total sleep time (TST)** | Sleep epochs × 0.5 min, on valid epochs |
| **Sleep efficiency** | 100 × TST / TIB in (i); 100 × TST / TIB_matched in (ii) |
| **Sleep onset** | First sleep epoch (any of codes 1–4), not first N2 |
| **Sleep onset latency (SOL)** | (onset of sleep onset epoch − onset of the first epoch of the recording) / 60, in minutes. A wall-clock difference, so it includes any human `8` and headband `-2` epochs before sleep onset (D8) |
| **Pre-sleep wake** | Wake epochs × 0.5 min before sleep onset, on valid epochs. 0 on a night that starts asleep. Equals SOL only when no `8` or `-2` falls before sleep onset |
| **REM latency** | (onset of first REM epoch − onset of sleep onset epoch) / 60, in minutes. From sleep onset, not from recording start |
| **REM latency, sustained** (sensitivity, D13) | Like REM latency, but to the **first epoch of the first run of at least 3 consecutive valid REM epochs** (1.5 min), counted from the sleep onset epoch. The rule is applied to both sequences alike. An excluded epoch breaks a run. NaN if the sequence has no such run, even when it has shorter REM stretches |
| **WASO** | Wake epochs × 0.5 min between sleep onset and the **last sleep epoch**, on valid epochs |
| **Terminal wake** | Wake epochs × 0.5 min after the last sleep epoch, to the end of the recording. Reported separately from WASO |
| **Stage percentage (N1, N2, N3, REM)** | 100 × epochs of the stage / TST. They sum to 100 %. Wake has no percentage (D9) |

Identity per night in (i), an exact partition of the valid epochs (human `8` removed):
`TIB = pre-sleep wake + TST + WASO + terminal wake + unscored`, where unscored is every headband `-2` epoch, before or after sleep onset.

Do **not** write it with SOL: SOL is a timestamp difference, so it also contains pre-onset `8` and `-2` epochs, which TIB excludes (`8`) or `unscored` already counts (`-2`). In general `SOL = pre-sleep wake + pre-onset 8 + pre-onset -2`. The identity is a unit test for `metrics.py`.

**Undefined values (D4, D5).** A value is NaN, never 0 and never the recording length, when:
- the target stage never occurs (REM latency with no REM; SOL with no sleep);
- the night is **left-censored** for that sequence, i.e. its first epoch is already sleep. Then SOL and REM latency are both NaN. Only sub-1 qualifies.

Pairwise statistics (Bland–Altman, correlation) use only nights where both values are defined. `results/clinical.json` reports, per metric, the nights used and the nights excluded with the reason.

**Flag, not exclusion:** a night is flagged "headband latency possibly delayed by dropout" when headband `-2` falls at or before the human target epoch (27 nights for sleep onset, 40 for first REM; `results/inventory_counts.json`). Sensitivity check without them.

**Sensitivity checks:** human `8` treated as Wake (D2); WASO including terminal wake (D11); sustained REM latency, a run of at least 3 REM epochs (D13).

## Agreement metrics (D1, D10)

| Metric | Definition |
|---|---|
| **(a) Bitbrain-comparable accuracy** | Agreeing epochs / epochs where the reference and the candidate are both 0–4. The sanity check against the dataset `README` |
| **(b) End-to-end accuracy** | Agreeing epochs / epochs where the reference is 0–4. A candidate `-2` counts as an error |
| **(c) Coverage** | Share of reference 0–4 epochs where the candidate gives a stage |
| **Kappa, per-stage precision / recall / F1** | Same epoch split as (a) and (b) |

**The three pairs (Milestone 3).** Each pair has a *reference* and a *candidate*:

| Pair | Reference | Candidate |
|---|---|---|
| 1 | human (`psg_hum`) | PSG AI (`psg_ai`) |
| 2 | human (`psg_hum`) | headband AI (`hb_ai`) |
| 3 | PSG AI (`psg_ai`) | headband AI (`hb_ai`) |

- **Human `8` epochs are excluded from every version of every pair, including pair 3.** The PSG was physically disconnected, so the PSG AI had no real signal there either (the PSG AI still gives a stage on part of those epochs), and it is not a meaningful reference. Excluding them everywhere also gives all three pairs the same base set of epochs, so their numbers are comparable.
- **Reference `-2` epochs are excluded, not counted as errors.** In pair 3 this means PSG AI `-2` epochs drop out of (a), (b) and (c): there is no reference label there. In pairs 1 and 2 the reference is the human label, which is never `-2`.
- **Candidate `-2` epochs** are skipped in (a), counted as errors in (b) and as missing in (c).
- Pair 3 is an **analysis tool only**: the PSG AI is not available on the device at runtime (README caveat).

**Implementation notes (`agreement.py`).**
- Per-stage scores use the five labels 0–4. A candidate `-2` is not one of them, so it lowers recall (a missed epoch) and leaves precision unchanged.
- Kappa is computed over all labels present, so in version (b) a candidate `-2` can never agree with the reference by chance. Its observed agreement equals the end-to-end accuracy.
- F1 is the count form 2TP / (2TP + FP + FN), from scikit-learn with `zero_division=np.nan`. It is NaN only when neither sequence has the stage, and 0 when the stage is present but entirely missed (or entirely false). It is **not** NaN just because precision or recall is: that would drop exactly the nights where the headband missed a stage completely and bias the F1 median upwards (revised 2026-10-26, Argon's sixth review).
- Confusion matrices have the reference stages 0–4 as rows. The columns are the candidate stages 0–4 and `-2`. Row-normalised matrices have NaN rows for a stage the reference never has.
- Per-night summaries report the median, the IQR and the number of nights where the value is defined. Output: `results/agreement.json`.

**Per-stage metrics per night (D10).** Recall is NaN if the reference has no such stage; precision is NaN if the candidate never predicts it; F1 is `2TP / (2TP + FP + FN)`, NaN only if neither sequence has the stage. Use `zero_division=np.nan` in scikit-learn, never `0`. Per-night kappa is NaN if a night has a single class in both sequences. Report, per stage, the number of nights where each metric is defined. The primary per-stage numbers are pooled over epochs.

## Clinical comparison (Milestone 5; `metrics.py`, `clinical.py`)

- **Difference** is always headband minus human, so a negative bias means the headband gives the smaller value.
- **Bland–Altman:** per night, over the nights where both values are defined: mean bias, SD of the differences, and the 95 % limits of agreement, bias ± 1.96 SD. Correlation is reported as Pearson's r and Spearman's ρ. Each metric also reports the median and IQR of the human values, the headband values and the differences, and how many nights were excluded and why (D5).
- **Per person (D7):** as a sensitivity, each person's nights are averaged first and the statistics are computed over people (`by_person`, `n` = people).
- **Left-censored** (D4): the recording's first epoch is valid and already sleep, in that sequence.
- **Sleep onset and the valid mask:** only valid epochs can be the sleep onset, the last sleep epoch or the first REM epoch. A headband sleep epoch during a human `8` is excluded entirely.
- **Matched-epoch analysis** drops nights with more than 20 % headband `-2`, measured as headband `-2` time divided by `TIB`. This selects the same 5 nights as checklist A3 (sub-1, 13, 57, 61, 77). The 10 % threshold and all nights are sensitivity checks.
- **Main figure:** Bland–Altman plots for sleep efficiency and WASO, chosen before looking at the results because they summarise the whole night and its wake component. Total sleep time shares sleep efficiency's denominator, and sleep onset latency is in the supplementary figure with every other metric.

## Hotspots: where human and headband disagree (D12; `hotspots.py`)

Reference: human (`psg_hum`). Candidate: headband AI (`hb_ai`). Human `8` epochs are excluded everywhere (D1). Distances are counted in epochs on the **full timeline** (D8): an epoch is 30 s, and rows are never dropped before measuring a distance.

**Disagreement rate** = 1 − accuracy over the epochs where the human and the headband both give a stage 0–4 (version (a) of D1). It measures staging disagreement only. A headband `-2` is not counted as a disagreement, because otherwise the "next to an artefact flag" analysis would be tautological: the flagged epochs themselves would be the errors. The `-2` effect is reported next to every rate as the **headband missing rate**: the share of human 0–4 epochs in the group where the headband gives `-2` (1 − coverage, D1 (c)).

**True stage.** The human stage (W, N1, N2, N3, REM) of the epoch. Also reported: where the disagreements go, as a row-normalised confusion matrix over version (a) epochs.

**Transition epoch (distance k = 0).** Epoch `i` is a transition epoch when its human label is a stage 0–4 and a neighbour (`i−1` or `i+1`) is a stage 0–4 with a **different** label. Both the last epoch of the old stage and the first epoch of the new stage are transition epochs. A boundary into or out of a human `8` is **not** a transition: a disconnection is not a sleep-stage change, and a change across a gap of `8`s is not counted either.

**Distance to the nearest transition.** For each epoch, the number of epochs to the closest transition epoch, in either direction. It is 0 on a transition epoch, 1 on its neighbours, and so on. Reported for k = 0…5, and one group "6+" for everything further, including nights with no transition. The same bins are used for the per-stage figure.

**Artefact flag.** A headband `-2` epoch. Only the headband's own flag is used, because that is the one available on the device (README caveat). For each epoch with a headband stage, two distances are taken on the full timeline:
- **before a flag:** epochs until the next flag (d = 1…5), meaning the epoch sits just before a flag;
- **after a flag:** epochs since the previous flag (d = 1…5), meaning the epoch sits just after one.

An epoch with a flag on both sides within 5 epochs counts in both groups. The reference group is "no flag within 5 epochs on either side". Flagged epochs are version (a) exclusions, so they never enter a rate.

**Time of night.** Hours since the recording start (`onset` / 3600, rounded down): hour 0 is the first hour, and so on. Nights run from about 3 to 9 h, so the late hours rest on fewer nights. All hours from 9 on are one group "9+". Every group reports its number of epochs and of nights.

**Per night and per person.**
- For every grouping, the pooled rate is accompanied by the **median and IQR over nights** (project rules). A night enters a group's per-night summary only when it has at least 10 epochs in that group, so a rate is never taken over a handful of epochs. Nights that have at least one version (a) epoch in the group but fewer than 10 are counted as undefined (`n_undefined`), not dropped silently, so `n_defined + n_undefined = n_nights` in every group.
- **Per participant:** the rate of each night, plus the mean of each person's nights (`pid`, D7), summarised over people.
- **Age and sex groups** use `pid` as the unit (D7). A person's rate in a group is the mean of that person's nights in the group, and the group statistic is the median and IQR over those person-level rates. Both `n_nights` and `n_pids` are reported. Age bands are 18–29, 30–44, 45–59 and 60+ (as in checklist A8); a night with age `n/a` is left out of the age groups and counted separately. Sex is `F` or `M` as in `participants.tsv`.

## Abstention flags (Milestone 6; `abstain.py`, D14)

A flag marks an epoch the user should not trust. **Device-only flags use the headband label sequence `hb_ai` and nothing else**, because a home user only wears the headband. Choosing the flag sets and the ranges of k below was done before looking at any result.

**Base set:** version (a) of D1: epochs where the human (reference) and the headband both give a stage 0–4, human `8` excluded. Flagged epochs are *abstained*; the rest are *kept*. With no flags, the kept epochs are the base set, so every curve starts at 100 % coverage with the Milestone 3 agreement for human vs headband (version (a)).

**Device-only flags** (all computed on the headband sequence, on the full timeline, per night):
- **`change_k`:** the epoch is within k epochs of a *predicted stage change*, for k = 0…5. A predicted change is the headband analogue of a human transition epoch (D12): the epoch has a headband stage 0–4 and a neighbour with a different stage 0–4. A boundary with `-2` is not a change. k = 0 flags the change epochs themselves.
- **`isolated`:** a single-epoch stage. The epoch has a stage 0–4, both neighbours exist and have stages 0–4, and both neighbours differ from it. The first and last epochs of the night are never flagged.
- **`artefact_k`:** the epoch is within k epochs of a headband `-2`, for k = 1…5. The `-2` epochs themselves are not in the base set.
- **Combination:** `combined_k` = `change_k` or `artefact_k`, for k = 0…5 (`combined_0` has no artefact part). "Isolated or `change_k`" would add nothing: an isolated epoch differs from both neighbours, so it is already a change epoch, and `isolated` is a subset of `change_0`.
- **`none`:** no epoch flagged. Every curve starts at this point.

**Upper-bound reference, not available on the device:** `psg_ai_disagrees` flags epochs where the PSG AI label differs from the headband label (a PSG AI `-2` counts as a difference). It needs the PSG, so a home user can't use it. It is a reference point for what a second opinion from the PSG could add, not a strict upper bound (a flag that marked exactly the human disagreements would do better), and it must be labelled as not available on the device wherever it appears.

**Chance reference:** abstaining on random epochs leaves the expected accuracy unchanged, so the reference line is the baseline accuracy.

**Metrics per flag set**
- **Coverage:** kept epochs / base epochs. Also reported against all human 0–4 epochs (`coverage_of_human_epochs`), where headband `-2` epochs count as already abstained.
- **Agreement:** accuracy on the kept epochs (human as reference). Also the accuracy on the flagged epochs, and **error capture**, the share of all disagreements that are flagged.
- **Per night:** the median and IQR over nights of the coverage and of the accuracy on kept epochs (nights with at least 10 kept epochs; others are counted as undefined).

**Clinical metric errors with flagged epochs removed or smoothed**
- **Excluded:** the kept epochs only, the same ones for both sequences. This is the matched-epoch analysis (D3 (ii)) with the flagged epochs also dropped: sleep efficiency, `TIB_matched` and the stage percentages are computed on the kept epochs. Operating points: `isolated`, `change_1`, `artefact_2`, `combined_1` and `psg_ai_disagrees`. The baseline is the same analysis with nothing flagged, on all nights. Both sequences are computed on the kept epochs only, so this measures staging agreement on those epochs. It does not say how well the whole-night value would be recovered from a subset of the night.
- **Smoothed:** every `isolated` epoch whose two neighbours carry the same label takes that label. The standalone clinical analysis (D3 (i)) is then recomputed on the smoothed headband sequence for every metric. A smoothed epoch keeps the same time in bed.
- **Error summary** per metric, over nights where both values are defined: mean bias, SD of the differences, width of the 95 % limits of agreement (3.92 SD) and the median absolute per-night error. The change is the value after minus the value before.

## Learned abstention from the labels alone (D16; `learned_abstain.py`)

Question, fixed before any result: can a small model, trained on the headband label sequence only, predict where the headband is wrong better than the hand-made flags of D14? No signal is read.

- **Target:** an epoch is an *error* when the human and the headband both give a stage 0–4 and the stages differ (the version (a) epochs of D1, human `8` excluded).
- **Features** come from the headband sequence `hb_ai` and the time since the recording start, never from the human or the PSG AI labels. Two sets:
  - **causal** (usable in real time): the label at the epoch and at the three before it (stage 0–4, `-2`, or "before the start"), epochs since the last predicted change, number of changes in the last 10 epochs, epochs since the last `-2`, share of `-2` in the last 20 epochs, hours since the start;
  - **non-causal** (a recording that is analysed afterwards): the labels at the epoch and three before and three after, epochs since the last and to the next predicted change, changes within five epochs either way, epochs since the last and to the next `-2`, share of `-2` within ten epochs either way, hours since the start.
- **Models:** a logistic regression and a gradient-boosted tree model (scikit-learn), each with default settings and no tuning.
- **Validation:** five folds that keep every night of one person together, so no person is in both the training and the test part. Every score is out of fold. Nothing is chosen on the test part.
- **Abstention:** epochs are ranked by the predicted probability of error. To keep a share *c* of the epochs, the *c* with the lowest probabilities are kept. Agreement on the kept epochs is the accuracy against the human.
- **Comparison with the hand-made flags:** at the coverage of each flag set of D14, the learned accuracy at that same coverage against the flag's accuracy. Also the area under the ROC curve (AUROC) and the average precision of the predicted probability of error, and, at the coverage of `change_1`, the per-night median and IQR of the coverage and of the accuracy on kept epochs.
- **Reference:** random abstention leaves the accuracy at the baseline. The PSG AI disagreement flag is shown as before, labelled as not available on the device.

## Signal quality of the headband (Phase 3; D17; `signal_quality.py`)

Questions about what the raw headband signal looks like when the headband gives no stage (`-2`), fixed before any systematic result. An informal first look at sub-1 had shown that most of that night's signal is exactly constant. The definitions below formalise that look, and the comparison is repeated on every complete night.

**Files.** Only complete files, read-only: the size and the SHA256 must match the git-annex key in `data/boas` (project rules). The recording start is the start of epoch 0 of the label files, and epoch *i* covers samples `[i * 30 * fs, (i + 1) * 30 * fs)`. Samples after the last whole epoch are ignored. The EDF can be a few seconds longer than the label file.

**Per epoch and channel (HB_1 and HB_2), in microvolts:**
- **`flat_share`:** the share of sample-to-sample steps inside the epoch that are exactly zero. 1 means a constant signal.
- **`std_uv`:** the standard deviation of the epoch.
- **`p2p_uv`:** maximum minus minimum.
- **`sat_share`:** the share of samples whose absolute value is at least 99 % of the converter's physical range limit.
- **`hf_ratio`:** power between 30 and 45 Hz divided by power between 0.5 and 45 Hz (Welch, 2 s windows, no overlap). Undefined when the total power is zero.
- **`mains_ratio`:** power between 49 and 51 Hz divided by power between 0.5 and 45 Hz. Undefined when the total power is zero.
- **Flat epoch:** an epoch where both channels have `flat_share` of at least 0.5.

**Q1. Is a `-2` epoch objectively bad signal?** For every complete night: the share of `-2` epochs that are flat epochs, the share of flat epochs that are `-2`, and the median and IQR of each feature in `-2` epochs against epochs that have a stage. Pooled over nights, the AUROC of each feature (and of `flat_share`) for telling `-2` epochs from staged epochs.

**Q2. Does movement come before dropout?** Only for nights whose file has the movement signals (`HB_IMU_1` to `HB_IMU_3` are the accelerometer, `HB_IMU_4` to `HB_IMU_6` the gyroscope, per the dataset README). A *long run* is at least 10 consecutive `-2` epochs. Movement of an epoch is the standard deviation of the accelerometer vector norm. For each long run, the mean movement over the 5 epochs before the run (if they are staged) against the same night's mean movement over staged epochs at least 30 epochs away from any `-2`. Reported: the number of runs, the median ratio and how many runs have a ratio above 1. **Revised 2026-10-27 after seeing the full data:** the ratio compares with the *mean* of a baseline that is full of movement spikes, so a quiet period reads well below 1 by construction. The fairer comparison is the **rank**: the share of five-epoch windows made only of staged epochs at least 30 epochs from any `-2` epoch whose mean movement is at most the mean of the five epochs before the run. A rank of 0.5 means typical movement. Reported: the median rank and how many runs have a rank above 0.5.

**Q3. Order of a person's nights.** From the EDF start date and time of the nights of people recorded more than once, compared with the order of the `sub-N` numbers (D15, C6).

**Q4. Does the signal differ when the headband does stage?** The same features for staged epochs near (within 5 epochs of) a `-2` epoch against staged epochs far from one (at least 30 epochs away).

## Three-class view for stimulation gating (D20; `three_class.py`)

A closed-loop stimulation system could use an online classifier with three outputs, wake, light sleep and deep sleep. The mapping from the five stages is **our assumption**: wake = W, light = N1, N2 and REM, deep = N3 (where REM belongs is not known, so a sensitivity run leaves REM out). Fixed before the results.

- **Base:** version (a) of D1: the human and the headband both give a stage 0–4, human `8` excluded.
- **Pooled:** the 3×3 confusion matrix (counts and row-normalised), accuracy, and for deep sleep precision, recall and F1, and for wake precision and recall.
- **Deep detected when it is not deep:** `1 − precision` of deep, and what the human scored in those epochs (N2, or another stage). This is the share of "headband says deep" epochs that are not N3 according to the human.
- **Deep missed:** `1 − recall` of deep. **End-to-end deep recall:** the same, with every human N3 epoch where the headband gives no stage (`-2`) counted as missed.
- **Per night:** deep recall over nights with at least 20 human N3 epochs, deep precision over nights with at least 20 headband-deep epochs (median and IQR, and the number of nights), and the number of nights with at least 20 human N3 epochs where the headband identifies none of them as deep.
- **Sensitivity:** all of the above with every epoch where either label is REM left out.

## One bad channel while the other is clean (Phase 3; D18; `one_channel.py`)

Question: when the headband gives no stage, is the signal really unusable, or is one of the two channels still clean? Thresholds fixed before the results, as proposed in the checklist, with a sensitivity run.

- **Reference amplitude** of a night and channel: the median `std_uv` (D17) over the epochs where the headband gives a stage and `std_uv` is above zero. A night without such epochs is skipped.
- **A channel is bad in an epoch** when its `flat_share` is at least 0.5, or its `std_uv` is at least *H* times the reference, or at most *L* times the reference. Primary: *H* = 10 and *L* = 0.1. Sensitivity: *H* = 5 and *L* = 0.2.
- **Epoch status:** *both good*, *one good* (exactly one channel bad), or *both bad*.
- **A. No-stage epochs.** For epochs where the headband gives no stage (`-2`): the share in each status, per night and pooled, and the minutes in *one good*.
- **B. One-good epochs.** The share of one-good epochs that have no headband stage, and which channel is the bad one. For one-good epochs where the headband **does** give a stage, its agreement with the human (epochs where the human also gives a stage 0 to 4), against the same agreement on both-good epochs.
- **C. Time that a clean channel could recover:** per night, the minutes of no-stage epochs that are one-good, as a share of the night's no-stage minutes. Median and IQR over nights with at least one no-stage epoch, and the pooled share.
- This analysis does not stage anything from one channel. It only counts how often a clean channel was available.

## What the headband signal looks like for each human stage (Phase 3; D19; `stage_signature.py`)

Exploratory questions, fixed before the results, about whether the raw headband signal carries the information that the human scoring implies, how the two channels relate, and what the amplitude spikes are. The human label comes from the PSG (which also has eye and chin channels), so this checks **consistency** between the headband signal and the human label. It cannot show that the human label is right.

- **Relative band power:** Welch power spectral density per epoch and channel (2 s windows, no overlap). Band power is the integral of the density over the band, and the relative power is the band power divided by the total power from 0.5 to 30 Hz. Bands, the conventional sleep EEG ones: delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–12 Hz, sigma 12–16 Hz, beta 16–30 Hz. Undefined (NaN) for a *constant* epoch, meaning one in which every sample is identical (a signal stuck at a constant level has no power, only floating-point rounding noise). A constant epoch is always a flat epoch in the sense of D17 (`flat_share` of at least 0.5 in both channels), but a flat epoch can still contain some signal and then has a defined spectrum.
- **Combined channel:** the mean of the relative powers of `HB_1` and `HB_2`. If one channel is undefined, the other one is used. If both are undefined, the epoch is undefined. Each channel is also analysed alone. No other way of joining channels is tried.
- **Epochs used:** the human gives a stage 0–4 (not 8) and the combined channel is defined.
- **Q1. Stage signature.** Per night and per channel (`HB_1`, `HB_2`, combined): the median relative power of each band within each human stage, over nights with at least 10 epochs of that stage. Reported: the median and IQR over nights.
- **Q2. Separation.** Per night, the area under the ROC curve (AUROC, 0.5 is chance, above 0.5 means higher in the first class) of each band, and of `log10(std_uv)`, for four contrasts: N3 against N2, N2 against N1, REM against N2, and W against all sleep stages (N1 to REM). Nights need at least 20 epochs in each class. Reported: the median over nights and the number of nights, for each channel alone and for the combined channel. Per-night AUROCs are used so that differences between nights do not count as separation.
- **Q3. Do the channels agree?** Per night, the Spearman correlation between `HB_1` and `HB_2` of `log10(std_uv)` and of relative delta power, over epochs where both are defined. Reported: the median over nights.
- **Q4. What are the amplitude spikes?** A *spike epoch* is an epoch with `HB_1` `std_uv` at least 3 times the night's median over live epochs. Reported: the share of spike epochs in each human stage against the share of all live epochs in that stage, and, for nights with movement sensors, the Spearman correlation between `log10(std_uv)` and the accelerometer movement of D17.

## Person effects (exploratory; `person_effects.py`, D15)

Three questions, fixed before any result was computed, about whether headband problems belong to the person, the night or the person's habits. **Exploratory:** few people are involved, several comparisons are made, and nothing is corrected for that.

**Inputs** (per night, from the results files): disagreement rate (D12) and headband dropout share (`hotspots.json`), human sleep efficiency, sleep onset latency, WASO and REM share (`clinical.json`); from `participants.tsv`: person (`pid`), sex, age, BMI. Dropout share is the share of human-scored epochs where the headband gives `-2`. People with at least two nights are the repeated-night group.

**A. Person or night?** Is headband dropout a property of the person (fit, anatomy, habits) or of the night (random slipping)?
- **ICC(1):** the one-way intraclass correlation of the dropout share over the repeated-night people, `(MSB − MSW) / (MSB + (n0 − 1) MSW)` with `n0` the average group size adjusted for unequal groups. Near 1: nights of one person resemble each other. Near 0 or below: no more than between different people. A 95 % interval comes from resampling people (fixed seed).
- **Spearman correlation over all pairs of nights of the same person.**
- A list of the high-dropout nights (more than 20 %, D3) that shows the same person's other nights.

**B. First nights against later nights.** Does anything change as people get used to the lab and the devices?
- **Order assumption:** the dataset has no dates (`scans.tsv` holds file names only), so a person's nights are ordered by the number in `sub-N`. This is an assumption. It is supported by one person's age increasing with the number, and it is not verified.
- Per person: *first* is the lowest-numbered night and *later* is the mean of the person's other nights. The difference is later minus first, over people with both values defined.
- Reported per metric: the median and IQR of the differences, how many people go up, down or stay the same, and a Wilcoxon signed-rank p-value for orientation only. Metrics: disagreement rate, dropout share, and the human sleep efficiency, sleep onset latency, WASO and REM share (the last four show whether the sleep itself changes between nights).

**C. Sex, BMI and age.** Person-level values, as in D7: a person's outcome is the mean over their nights (dropout share, disagreement rate), and a person's BMI and age are means over the nights with a value. People without a value are left out and counted.
- **Sex:** median and IQR per group, with the number of people, and a Mann–Whitney p-value.
- **BMI and age:** Spearman correlation with the number of people and its p-value.

**p-values** are orientation only. With this many comparisons and these sample sizes, a small p-value is not evidence.

## Summary statistics (D7)

Distributions are over **nights** (median, IQR). Group comparisons and any inferential claim use **`pid`** as the unit (average each person's nights, or bootstrap people). Always state both n's.
