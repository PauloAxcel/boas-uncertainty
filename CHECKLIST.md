# Open questions checklist

The single place where questions about the data and analysis decisions are tracked. When a question is settled, write the answer under it, change its status, and note the date and who decided.

**Status**
- `[ ]` open
- `[~]` plausible answer written, waiting for confirmation or a decision
- `[x]` resolved

**Sources** (where a question came from)
- `INV-n`: `docs/data_inventory.md`, section 9, question n
- `ARGON`: Argon's reviews of the inventory, 2026-10-24 and 2026-10-26
- `CLAUDE`: raised while writing this checklist

Unless stated otherwise, every number below was recomputed from `data/boas` on 2026-10-24 with the loader in `src/boas_unc/inventory_figures.py`.

**The short version:** twelve decisions block Milestones 3–5 (D12 covers the Milestone 4 hotspots). D1 and D10 cover agreement; D2–D6, D8, D9 and D11 cover the clinical metrics; D7 covers per-night vs per-person statistics. **D1–D12 were locked on 2026-10-26** after Argon's third review, and are written up in `docs/definitions.md`. H5 is done (Milestone 2).

---

## A. Facts about the data

These can be answered from the files.

### A1. `[x]` Do Bitbrain's headline accuracies use the current labels, and how do they treat codes 8 and -2? (INV-7, INV-2/3, ARGON)

**Answer:** Yes, they use the current labels, and they **drop every epoch where either label is a special code** (8 or -2). Pooled accuracy over epochs where both labels are stages 0–4:

| Pair | Agreeing / epochs | Accuracy | `README` |
|---|---|---:|---:|
| human vs PSG AI | 104 019 / 119 389 | **87.13 %** | 87.13 % |
| human vs headband AI | 99 654 / 114 890 | **86.74 %** | 86.71 % |

- The PSG figure matches exactly.
- The headband figure is 0.03 points off. That fits `CHANGES` 1.1.1 ("fewer than 0.3% of the labels were affected"), so the current labels are the ones to use.
- Argon got the same percentages but slightly different counts (103 995 / 119 361 and 99 669 / 114 907). The counts above come from the repo code. `agreement.py` must reproduce them in Milestone 3.

The same agreement computed other ways (feeds into D1):
- All 120 095 epochs, special codes counted as mismatches: 86.61 % (PSG AI), 82.98 % (headband AI).
- Epochs where the human label is 0–4 (119 759 epochs), with an AI `-2` counted as an error: 86.86 % (PSG AI), 83.21 % (headband AI).

### A2. `[~]` Why does sub-1 start in N3? (INV-4, ARGON)

**Most plausible answer:** the recording started after the participant had already fallen asleep, or the start was cropped. It isn't a human labelling error.

Evidence: all three sources score N3 from epoch 0.

```
psg_hum  3333333333333333333333333333333333333222   (first 40 epochs)
psg_ai   3333333333333333333333333333333333333322
hb_ai    33333333xxxxxxx3333333333333333333333222   (x = -2)
```

Caveat: the AI was trained on human labels, so its agreement isn't fully independent evidence. Only Bitbrain can confirm when the recording started.

Consequence: see D4.

### A3. `[~]` Why is the headband `-2` concentrated at the end of the night and in a few nights? (INV-16, ARGON)

Facts:
- 2 928 of the 4 881 headband `-2` epochs (60 %) are in runs that last until the end of the recording.
- **Six nights hold 49.3 % of all headband `-2` epochs** (2 407 / 4 881):

  | Night | `-2` epochs / night length | % of night |
  |---|---|---:|
  | sub-1 | 677 / 915 | 74.0 % |
  | sub-13 | 544 / 997 | 54.6 % |
  | sub-57 | 451 / 1 032 | 43.7 % |
  | sub-61 | 266 / 948 | 28.1 % |
  | sub-77 | 265 / 1 078 | 24.6 % |
  | sub-55 | 204 / 1 073 | 19.0 % |

- sub-1 is the only night with no headband REM at all. The human scorers found 87 REM epochs that night.
- The headband `-2` share per night:
  - Over all 128 nights: median 0.23 %, IQR 0.10–3.14 %.
  - Over the 100 nights that have any `-2`: median 0.72 %, IQR 0.12–5.02 %.
- Argon reported a median of 0.49 % (IQR 0.10–2.26 %). **That doesn't reproduce** under either denominator above. Use the values from the code.
- Number of nights by headband `-2` share:
  - more than 5 %: 25 nights
  - more than 10 %: 12 nights (sub-1, 7, 11, 13, 29, 55, 57, 61, 65, 76, 77, 88)
  - more than 20 %: 5 nights (sub-1, 13, 57, 61, 77)

- **During those 2 928 trailing epochs the human scorers still had the person asleep in 82.0 % of them** (2 402: 100 N1, 1 756 N2, 127 N3, 419 REM; 525 Wake; 1 code `8`). Source: `results/inventory_counts.json`, `hb_ai_minus2_end_of_night_runs_human_stages`.

**Most plausible explanation:** in the tail runs, the headband dropped out while the person was mostly still asleep (82 %, above), rather than being taken off on waking. The long mid-night blocks (e.g. sub-13, sub-57) are probably the electrodes losing contact. **Not confirmed**: the EDF signals would show which (Phase 3).

Consequence: see D3.

### A4. `[x]` What units is `RecordingDuration` in? (INV-13)

**Answer:** seconds. That's the BIDS definition, and `RecordingDuration − 30 × n_epochs` falls in [0, 29] for every night.

### A5. `[x]` Can `EEGChannelCount` be trusted? (INV-10)

**Answer:** no. It counts the rows typed `EEG` in `*_channels.tsv`, and those rows include EOG, EMG and breathing channels. Use `*_channels.tsv` as the source of truth.

### A6. `[x]` Are there lights-off or lights-on markers? (INV-5)

**Answer:** no. The events files contain only the epoch timing columns and the stage columns (inventory section 3), and no sidecar mentions such markers. Time in bed has to be the full recording. See D6.

### A7. `[~]` Is pid 15 the same person on both nights? (INV-14)

**Most plausible answer:** yes, recorded about a year apart. Age goes from 34 to 35, sex is M on both, BMI goes from 28.1 to 27.3, and the recording numbers are far apart (sub-23 vs sub-124). `participants.json` defines `pid` as the same for all sessions of one person, so group by `pid`.

### A8. `[~]` Why is there so little N3? (INV-15, ARGON)

**Most plausible answer:** mostly age, plus a lower baseline even in young adults, which only Bitbrain can explain (C5).

Human N3 **as a % of sleep (N1–REM)**, per night:
- Median 2.03 %, IQR 0.00–8.07 %. The inventory's earlier "1.65 %" used all recorded epochs as the denominator; see D9.
- N3 falls with age: Spearman ρ = −0.51 across the 127 nights with a known age.

| Age | Nights | Median N3 % | Mean N3 % | Nights with no N3 |
|---|---:|---:|---:|---:|
| 18–29 | 54 | 7.73 | 8.10 | 5 |
| 30–44 | 27 | 4.13 | 5.58 | 6 |
| 45–59 | 8 | 0.16 | 2.30 | 4 |
| 60+ | 38 | 0.00 | 1.81 | 22 |

Source: `results/inventory_counts.json`, `human_N3_pct_of_sleep_by_age_band`.

- **Age accounts for most of the nights with no N3, but not all.** 22 of the 37 such nights are from people aged 60+. Argon's review said age "explains the 37" outright, which overstates it. The other 15 are spread across younger ages.
- **Young adults are also low.** The 18–29 median of 7.7 % is about half the textbook ~13–23 %. That points to the scoring practice or this population, and only Bitbrain can say which (C5).

Consequences:
- Pooled N3 statistics rest on few epochs.
- Per-night N3 metrics are undefined on many nights: 37 nights have no human N3 and 56 have no headband N3. See D10.
- Age is a necessary grouping variable in Milestone 4.

---

## B. Decisions to make before Milestones 3–5

Each decision has a proposed answer. Once it's agreed, the chosen rule goes into `docs/definitions.md`, as the project rules require.

### D1. `[x]` Which epochs count when measuring agreement? (`agreement.py`; INV-2/3, ARGON)

**Proposed:** report three versions.

| Version | Epochs counted | Headband AI accuracy | Purpose |
|---|---|---:|---|
| **(a) Bitbrain-comparable** | both labels 0–4 | 86.74 % | Sanity check against `README` |
| **(b) End-to-end** | human 0–4; AI `-2` counted as an error | 83.21 % | What a user actually gets |
| **(c) Coverage** | share of human 0–4 epochs where the AI gives a stage | — | Reported next to (a) |

- Human `8` epochs are excluded from every version, because there's no reference label to compare against.
- Argon's "all 120 095 epochs" version counts human `8` as an AI error. That's unfair to the AI, because the humans had no signal either.
- Kappa and per-stage F1 follow the same split.

**Pair 3 (PSG AI vs headband AI), locked 2026-10-26 after Argon's fifth review:** the three versions apply to all three pairs, with a *reference* and a *candidate* (`docs/definitions.md`). In pair 3 the PSG AI is the reference. Human `8` epochs are excluded in every pair, so the pairs share one base set of epochs. PSG AI `-2` epochs are excluded in pair 3 (no reference label), while headband `-2` counts as an error in (b). Keeping the `8` epochs instead would change pair 3 only in the second decimal, but would mix in epochs where the PSG was unplugged. The exact numbers for each choice come from `agreement.py` into `results/agreement.json`.

### D2. `[x]` How do human `8` epochs (PSG disconnected) enter the sleep metrics? (`metrics.py`; INV-2, ARGON)

**Locked 2026-10-26:** treat them as **unscored time**, and **keep them on the timeline**.
- `load.py` does **not** drop `8` rows. Every epoch stays, with boolean flags (Milestone 2, D8), so timestamp-based latencies and the k-epoch transition windows of Milestones 4 and 6 stay intact.
- The exclusion happens when metrics are computed: `8` epochs are left out of time in bed, total sleep time, WASO and terminal wake, **for both** label sequences, on the same epochs, so the human-vs-headband comparison stays like-for-like.
- `metrics.py` stays pure: the caller passes the sequence together with a boolean mask of valid epochs. The functions never look for code `8` themselves.
- Sensitivity check: rerun with `8` treated as Wake.

The data supports either choice: during these gaps the headband says Wake 310 times out of 336. Affected: 336 epochs (2.8 h in total) across 37 nights.

### D3. `[x]` How do headband `-2` epochs enter the sleep metrics, and should nights with heavy missingness be excluded? (`metrics.py`; INV-3, INV-16, ARGON)

**Revised 2026-10-24 after Argon's second review; locked 2026-10-26 with the TIB / TIB_matched split.** The first version contradicted itself: item 1 deleted every `-2` epoch from both sequences, so item 4's trimming of `-2` tails had nothing left to act on. It also cut real, human-scored sleep out of the PSG reference, which made headband sleep-time errors look smaller than they are. For example, a 1.5 h headband dropout during sleep would remove 1.5 h of true sleep from the reference too.

**Proposed:** compute two analyses and report both, because each answers a different question.

| Analysis | Epochs used | Question it answers |
|---|---|---|
| **(i) Standalone (primary)** | Each sequence on its own full timeline. The human reference uses every human-scored epoch; D2 handles code `8`. Headband `-2` counts as **unscored**: neither sleep nor wake | "What would a clinician read off the headband report?" Dropout shows up as lost sleep time, which is the real cost of the device |
| **(ii) Matched-epoch** | Only epochs where both sequences have a stage 0–4 | "When the headband does give a stage, how much does its staging alone shift the metrics?" |

- The gap between (i) and (ii) splits the headband's error into a **dropout part** and a **staging part**.
- **Time in bed differs between the analyses.** (i) uses `TIB` (D6). (ii) uses `TIB_matched`: the number of epochs, times 30 s, where both sequences have a stage 0–4. Otherwise dropout would shrink sleep efficiency in (ii) by the dropout rate for both sequences, and (ii) would stop isolating staging disagreement. Within one analysis, both sequences share the same denominator.
- Latencies, WASO and terminal wake are only computed in (i); see D8.
- Headband missingness per night (median, IQR, and the per-night list from A3) is reported as a result in its own right.
- **Night exclusion:** (i) keeps all nights, because heavy dropout is exactly the effect being measured, and flags the high-dropout nights. (ii) excludes nights with more than 20 % headband `-2` (5 nights), because metrics computed on a fraction of a night aren't comparable. The 10–20 % group is 7 more nights: sub-55 19.0 %, sub-11 16.5 %, sub-7 16.3 %, sub-88 15.2 %, sub-29 13.6 %, sub-65 13.2 %, sub-76 13.1 %. Sensitivity checks: a 10 % threshold, and keeping all nights.
- Decide the threshold before looking at the clinical results.

### D4. `[x]` Sleep onset latency and other latencies for sub-1 (and any night that starts asleep) (`metrics.py`; ARGON)

**Proposed:**
- A night whose first epoch is already sleep in a given sequence is **left-censored** for that sequence. Set **both sleep onset latency and REM latency to NaN**, not 0 (REM latency is measured from sleep onset, which is unknown), and flag the night.
- The recomputed check: sub-1 is the only such night. The 6 nights that start with 1–2 headband `-2` epochs all go straight to Wake in both sequences, so they are not censored.
- sub-1 also has 74 % headband `-2`. It stays in the standalone analysis as a flagged night, and the matched analysis excludes it (D3).
- Keep sub-1 in the agreement analysis, where it's an ordinary set of epochs.

### D5. `[x]` What happens when a latency is undefined (e.g. REM never occurs)? (`metrics.py`; ARGON)

**Proposed:**
- A latency is NaN when the stage never occurs.
- Bland–Altman plots and statistics use only nights where both values are defined.
- `results/clinical.json` reports, per metric, the number of nights used and the number excluded and why.
- Never fill in a value or substitute the recording length.

### D6. `[x]` How is time in bed defined? (INV-5, README Milestone 5)

**Proposed:** from the start to the end of the recording, minus human `8` epochs (D2). The same time in bed (`TIB`) applies to the human and headband sequences in the standalone analysis (D3 (i)). The matched analysis uses `TIB_matched` (D3 (ii)).
- Headband `-2` epochs, including the trailing runs, stay inside time in bed and count as unscored (D3). This replaces the earlier "trim headband tails" idea, which would have given the headband a different denominator from the reference. The A3 check supports it: the person was asleep in 82 % of the trailing `-2` epochs.
- This is the default `README` suggests when the files have no lights markers, which A6 confirmed.

### D7. `[x]` Should statistics be per night or per person? (`hotspots.py`, `REPORT.md`; ARGON)

**Proposed:**
- Distributions are reported over **nights**, as the project rules ask: 128 nights, or fewer after the D3 exclusions in the matched analysis.
- Anything that compares groups or makes an inferential claim (age or sex groups, confidence intervals) treats **`pid` as the unit**: average each person's nights, or use a bootstrap that resamples people rather than nights.
- Always state both n's.

The numbers: 20 of the 100 participants contribute 48 of the 128 nights (12 people with 2 nights, 8 with 3).

### D8. `[x]` How are latencies computed when some epochs are special codes? (`metrics.py`; ARGON)

**Proposed:**
- Compute latencies as **differences of `onset` timestamps** on the full, unfiltered timeline: (onset of the target epoch − onset of the reference epoch) / 60, in minutes.
- **Never count rows after dropping `8` or `-2`.** Dropping rows shortens the timeline and would shift every latency earlier. This is why `load.py` keeps every epoch (D2).
- A latency that starts or ends in a left-censored night is NaN (D4).
- Latencies are computed in the standalone analysis only (D3 (i)). In the matched analysis, the masking changes which epoch counts as "first", so latencies aren't defined there.

The problem is not rare:
- 27 nights have a headband `-2` at or before human sleep onset. On sub-61, the human sleep-onset epoch itself is `-2` on the headband.
- 40 nights have a headband `-2` at or before the first human REM epoch. On sub-1, sub-57 and sub-61, the first human REM epoch itself is `-2` on the headband.

Source: `results/inventory_counts.json`.

Human `8` also sits before the target epoch on some nights: 6 nights before human sleep onset and 15 before the first human REM epoch (`nights_human_8_before_human_sleep_onset`, `nights_human_8_before_first_human_REM` in `results/inventory_counts.json`). Those minutes count as part of the latency, because the latency is a timestamp difference.

Flag these nights as "headband latency possibly delayed by dropout" and run a sensitivity check without them.

### D9. `[x]` What denominator do stage percentages use? (`metrics.py`, `docs/definitions.md`; ARGON)

**Proposed:**
- Follow AASM: % N1, % N2, % N3 and % REM are each **a share of total sleep time** (epochs of stages 1–4 in that sequence). They sum to 100 %.
- Wake isn't given as a percentage. Sleep efficiency, sleep onset latency and WASO cover it.

This matters because two denominators had been mixed:
- Figure 2 and the inventory's "median 1.65 % N3" use **all recorded epochs**. That's fine for describing the night, but it isn't a clinical stage percentage. On nights with heavy headband `-2`, the headband's percentages sum to well under 100 %.
- As a % of sleep, the median human N3 is 2.03 % (A8).

Figure 2 is now labelled "% of all recorded epochs (not % of sleep)".

### D10. `[x]` What happens to per-night, per-stage agreement metrics when a stage never occurs? (`agreement.py`; ARGON)

**Proposed:**
- **Primary:** per-stage precision, recall and F1 **pooled over epochs**, where every stage has support.
- **Per night:**
  - Recall is NaN when the human labels don't contain the stage.
  - Precision is NaN when the headband doesn't predict it.
  - F1 is the count form `2TP / (2TP + FP + FN)`: NaN only when neither sequence has the stage. **Revised 2026-10-26 (Argon's sixth review).** The first version made F1 NaN whenever precision or recall was NaN. That dropped the nights where the headband missed a stage completely (recall 0) while keeping the ones where it hit a little, so the per-night N3 F1 median looked much better than the N3 recall median. With the count form the headband's N3 F1 is defined on 98 nights, not 65 (`results/agreement.json`, `human_vs_hb_ai`, `b_end_to_end`, `per_night`, `N3_f1`).
  - With scikit-learn, use `zero_division=np.nan`. **Never use `zero_division=0`**: that would score a night with no N3 at all as a total N3 failure.
- Report, per stage, the number of nights where each metric is defined. For N3 that's at most 91 nights for recall, 72 for precision and 98 for F1.
- Per-night kappa is NaN only if a night has a single class in both sequences. That doesn't happen here, but guard for it anyway.

### D11. `[x]` WASO, terminal wake and the sleep-period identity (`metrics.py`; ARGON)

**Locked 2026-10-26:**
- **Sleep onset** is the first epoch of any sleep stage (N1, N2, N3 or REM; codes 1–4), not the first N2.
- **REM latency** is measured from sleep onset: (onset of first REM − onset of sleep onset) / 60, in minutes.
- **WASO** counts Wake epochs between sleep onset and the **last sleep epoch**.
- **Terminal wake** counts Wake epochs after the last sleep epoch, to the end of the recording. It is reported separately.
- Identity per night: `TIB = pre-sleep wake + TST + WASO + terminal wake + unscored`, where unscored is every headband `-2` epoch and pre-sleep wake is Wake before sleep onset (0 on a left-censored night). **Corrected 2026-10-26 after Argon's fourth review:** the earlier version used SOL, which is a timestamp difference that also contains pre-onset `8` and `-2` epochs, so it double-counted them and failed on some nights (and gave NaN on sub-1). Checked on all 128 nights for both sequences: the new form holds exactly, the old one does not.

Why WASO stops at the last sleep epoch: with no "lights on" marker (A6), trailing Wake mostly means the recording kept running after the person woke up. Counting it as WASO would make WASO depend on when someone stopped the recording. The data: 34 of 128 nights have more than 5 min of terminal wake (human reference; median 1.5 min, max 170.5 min). Source: `results/inventory_counts.json`, `human_terminal_wake_min_per_night_q25_median_q75_max` and `nights_human_terminal_wake_over_5_min`.

Sensitivity check: also report WASO including terminal wake.

### D12. `[x]` How are the Milestone 4 hotspots defined? (`hotspots.py`; ARGON)

**Locked 2026-10-26** (full text in `docs/definitions.md`, section "Hotspots"):
- **Transition epoch (k = 0):** a stage 0–4 epoch with a neighbour that is a *different* stage 0–4. A boundary into or out of human `8` is **not** a transition, and neither is a change across a gap of `8`s. Distance is counted in epochs on the full timeline, k = 0…5 and "6+".
- **Rate denominator:** version (a) of D1, i.e. epochs where human and headband both give a stage 0–4. Headband `-2` is **not** counted as a disagreement, so that "next to an artefact flag" isn't tautological. The `-2` effect is reported beside every rate as the headband missing rate.
- **Artefact flags:** the headband's own `-2` only, because it is the one available on the device. Epochs 1…5 before and after a flag, against epochs with no flag within 5.
- **Time of night:** whole hours since the recording start, hours 0–8 and "9+". Every group reports its epochs and nights.
- **Groups:** age bands (A8 bands) and sex use `pid` as the unit (D7). Both `n_nights` and `n_pids` are reported. The one night with age `n/a` is left out of the age groups.
- A night enters a per-night summary only with at least 10 epochs in the group.

### D13. `[x]` Is REM latency measured to the first REM epoch, or to the first sustained REM run? (`metrics.py`; found in Milestone 5)

**Locked 2026-10-26** (Argon's seventh review agreed with the proposal):
- **Primary:** the first REM epoch, the plain reading of "REM latency" (D11). The raw headband staging gives a REM latency about an hour shorter than the human's (bias −57.8 min).
- **Sensitivity:** the first run of at least 3 consecutive valid REM epochs (1.5 min), counted from the same sleep onset epoch and applied to **both** sequences. An excluded epoch breaks a run. A sequence with REM epochs but no run of 3 gets NaN, and `bland_altman` reports the nights used and excluded in the same format as the primary metric. The threshold of 3 was fixed before the result was computed.

**Result** (`results/clinical.json`, `sensitivity.rem_latency_sustained_run_of_3`, 127 nights, none lost to the new rule):

| | Mean bias (min) | 95 % limits of agreement | Median per-night difference (IQR) | Median human / headband (min) |
|---|---:|---|---|---:|
| First REM epoch | −57.8 | −207 to +92 | −31.5 (−101.8 to −0.5) | 102 / 60 |
| First run of ≥ 3 epochs | −40.2 | −194 to +114 | −3.0 (−81.8 to +0.5) | 115 / 79.5 |

The mean and the median tell different stories, so report both (the project rules ask for per-night distributions):
- **The typical night is fixed by the 3-epoch rule.** The median difference goes from −31.5 to −3.0 min, so on a typical night the headband's sustained REM latency is within a few minutes of the human's.
- **The mean is dragged by a tail of nights.** 37 of 127 nights (29 %) still have the headband more than 60 min early under the ≥ 3 rule (54 under the first-epoch rule), and 43 (34 %) are more than 30 min early. Only 3 nights go the other way by more than 30 min. Nights within ±3.5 min of the human: 54 of 127 (43 %) under the ≥ 3 rule, 43 (34 %) under the first-epoch rule.
- So brief early REM predictions explain most of the gap on a typical night, but not the long tail. On the tail nights the headband reaches a sustained REM stretch much earlier than the humans do. Those stretches may be false, or the human scorers may have called the same epochs N2 or N1: the labels alone can't say which (Phase 3, with the signals).

**Why it happens** (observed, not yet confirmed): the headband's first REM stretch is short (median 5 epochs against 28 for the human). 21 of 127 nights have a headband first REM run of one epoch, and 10 have one in the human labels.

### D14. `[x]` Which abstention flags, and how are agreement and clinical errors measured? (`abstain.py`; README Milestone 6)

**Locked 2026-10-26**, before any abstention result was computed (full text in `docs/definitions.md`, "Abstention flags"):
- **Device-only flags** from the headband sequence alone: within k epochs of a predicted change (k = 0…5), isolated single-epoch stage, within k epochs of a `-2` (k = 1…5), and the combination `combined_k` (change or artefact). `isolated` is a subset of `change_0`, so "isolated or change" is not a separate set.
- **Base set** is D1 version (a), so the curves start at 100 % coverage with the Milestone 3 agreement.
- **Upper bound, not on the device:** `psg_ai_disagrees`. **Chance reference:** random abstention leaves accuracy at the baseline.
- **Clinical effect:** excluded (kept epochs only, matched-epoch style) and smoothed (isolated epochs with equal neighbours take their label). Errors are bias, SD, limits-of-agreement width and the median absolute per-night error.

**Results** (`results/abstention.json`, `results/agreement.json` baseline of 86.74 % reproduced exactly):
- Agreement on kept epochs rises as more is flagged: `change_1` gives 92.1 % at 82.2 % coverage, `change_3` 93.6 % at 72.3 %, `change_5` 94.4 % at 64.5 %. The gain comes almost entirely from the predicted-change flag.
- `isolated` alone gives 87.7 % at 97.4 %, and the artefact flags barely move agreement (86.8–86.9 %), because the epochs next to a `-2` are few.
- The PSG AI disagreement flag (not on the device) gives 92.3 % at 88.9 % coverage, a little above the device-only `change_0` (90.6 % at 88.7 %). So a device-only heuristic recovers most of what a PSG second opinion offers at that coverage.
- **Smoothing** isolated epochs changes 1 % of the epochs and leaves agreement at 86.74 % (kappa 0.769 to 0.769). It helps some wake and latency metrics (sleep onset latency bias 2.1 to 0.6 min, WASO 1.6 to 0.2 min) and leaves the rest roughly as they were.
- **Excluding** epochs next to a predicted change roughly halves the sleep-efficiency error on the kept epochs (median absolute error 1.23 to 0.61 points) and trims the stage-percentage errors, but it removes about 18 % of the time in bed (449 to 369 min on average). This measures agreement on the kept epochs, not recovery of the whole-night value.

### D15. `[x]` Is headband dropout a person effect or a night effect, and do sex, BMI and age matter? (`person_effects.py`; asked 2026-10-27)

**Locked 2026-10-27**, before any result was computed (full text in `docs/definitions.md`, "Person effects"). Three exploratory questions: (A) are the dropout shares of one person's nights alike (ICC, pair correlation), (B) do first and later nights differ (order by `sub` number, which is an assumption because there are no dates), (C) does person-level dropout or disagreement relate to sex, BMI or age. p-values are for orientation only.

**Results** (`results/person_effects.json`, `figures/person_effects/fig15_person_effects.png`; 20 people with 48 nights, 12 with two nights and 8 with three):
- **A. Person or night?** The ICC of the dropout share is 0.20, with a 95 % interval of −0.01 to 0.63, and the correlation between nights of the same person is −0.01 over 36 pairs. The interval is wide, so a moderate person effect can't be ruled out, but the data don't support one. Of the 5 nights with more than 20 % dropout, 4 belong to people with another night: the other nights have 0.8 %, 0.0 %, 6.2 % and 19.0 % dropout. The 19.0 % night (sub-55, same person as sub-57) is the one case where both of a person's nights are affected. The rest look like single bad nights.
- **B. First against later nights** (order assumed, C6): nothing clear. Human sleep efficiency goes up for 10 people and down for 10 (median difference +0.5 points). Dropout goes up for 13 and down for 6 (p = 0.12), so there is no sign that it falls as people get used to the band. The p-values are all above 0.1.
- **C. Sex, BMI, age** (100 people): dropout is higher in women than men (median 0.91 % against 0.26 %, p = 0.09), which is the opposite direction to the idea that women cope better with head-worn devices, and it is not clear. BMI shows no relation to dropout (ρ = −0.03) or to disagreement (ρ = −0.01). Age shows no relation to dropout (ρ = 0.06), but older people have a lower disagreement rate (ρ = −0.25, p = 0.013, 99 people). That p-value would not survive a correction for the number of comparisons made, and I haven't looked for a reason.

### D16. `[x]` Can a model trained on the headband labels alone beat the hand-made abstention flags? (`learned_abstain.py`; asked 2026-10-27)

**Locked 2026-10-27**, before any result was computed (full text in `docs/definitions.md`, "Learned abstention"). Causal and non-causal feature sets, two untuned models, five person-grouped folds, comparison at matched coverage with the D14 flags.

**Results** (`results/learned_abstention.json`, `figures/abstention/fig16_learned_vs_heuristic.png`): the label-only information is mostly used up by the simple flags.
- **Prediction of errors (AUROC):** hand-made flags 0.65 (`change_0`) to 0.72 (`change_3`). Gradient boosting, causal 0.76, non-causal 0.80. Logistic regression is lower (0.72 and 0.77).
- **At the coverage of the hand-made flags:** the non-causal gradient-boosted model is 0.4 to 0.6 points more accurate (for example 92.5 % against 92.1 % at the coverage of `change_1`, and 94.2 % against 93.6 % at that of `change_3`). The causal model, which only sees labels up to the epoch (usable in real time), is level with the hand-made flags at `change_0` and `change_1` (within 0.1 points) and 0.3 to 0.8 points behind them at larger ranges.
- **The PSG reference still wins:** the PSG AI disagreement flag gives 92.3 % at 88.9 % coverage, against 90.9 % for the best label-only model at that coverage. A second opinion from the PSG holds information that the headband's own labels do not.
- **Reading:** a model that is allowed to see more of the label sequence adds a fraction of a point. Where the headband is wrong cannot be predicted much better from its labels alone, which supports Phase 3: the next gain needs information from the signal, such as the model's probabilities and signal quality.


### D17. `[x]` What does the raw headband signal look like when the headband gives no stage, and does movement come first? (`signal_quality.py`; Phase 3, started 2026-10-27)

**Locked 2026-10-27** (full text in `docs/definitions.md`, "Signal quality of the headband"), after an informal first look at sub-1 that showed most of its signal exactly constant. Q1 `-2` against signal quality, Q2 movement before long dropout runs (nights with the movement sensors only), Q3 order of nights from EDF dates, Q4 signal near `-2`. Data rule: complete files only, size and SHA256 checked, read-only, from the SD-card folder.

**Data facts** (from the git-annex keys, `data/boas`): the 128 headband EDFs total 13.44 GB, between 26 MB and 155 MB each (median 124 MB). sub-1 is the smallest. An earlier estimate of about 3.6 GB, extrapolated from sub-1, was wrong. EDF headers hold the recording start date and time.

**Results on all 128 nights** (complete files only, `results/signal_quality.json`; `uv run python -m boas_unc.signal_quality`, plots with `uv run python -m boas_unc.plot_night sub-N`). **Several statements made on the first 15 nights were wrong or too strong, and are corrected here.**
- **What is behind a no-stage epoch.** 4,881 epochs in 100 nights get no headband stage. In 2,755 of them (56.4 %) both channels are flat. The rest are mostly one channel wild or near-silent while the other is clean (D18).
- **Correction: a flat signal is not always flagged.** On 15 nights every flat epoch had no stage. Over all nights, 315 of the 3,070 flat epochs (10.3 %) do have a headband stage, in 3 nights: sub-96 (237 epochs), sub-82 (50) and sub-44 (28). In sub-44 the signal is completely flat for the last 28 epochs and the headband labels every one of them N2 (the human scored REM). In sub-82 and sub-96 the flat share is mostly 0.5 to 0.8, so part of each epoch has signal. Across the three nights the headband agrees with the human on 172 of 315 epochs (54.6 %).
- **Kind 1, a dead (railed) signal.** The signal sits at one constant value, −2999.9 µV, identically in both channels: the bottom of the file's export range (the amplifier input range is ±100 mV). On sub-1 the signal looks normal (tens of µV) until one second before a large transient, then steps to the rail and stays for the last 669 epochs (about 5.6 hours). The EDF is as long as the label file, so the amplifier kept sampling. This favours a sudden loss of electrode contact over a stopped recording (an inference). On sub-3 and sub-11 the final flat stretch begins with a huge movement spike, which looks like the band being taken off.
- **Kind 2, one bad channel while the other is clean** (sub-7, sub-11 and others, D18).
- **Q2, movement before long dropouts** (94 nights with the movement sensor, 46 stretches of at least five minutes with no stage that follow staged epochs). The ratio of movement just before to the mean of a baseline has a median of 0.30, but that comparison is biased (a baseline mean full of spikes), so it was replaced by a rank against the same night's other five-epoch windows: **median rank 0.61 (IQR 0.24 to 0.87), above 0.5 in 25 of 46 stretches.** So movement shortly before a dropout is only slightly above typical, and varies a lot. It does not show that gross movement causes most dropouts.
- **Q3, order of nights (resolves C6).** For all 20 people with more than one night, the order of the EDF start dates equals the order of the `sub-N` numbers, so the first-against-later analysis (D15) did use the real order.
- **Pooled AUROC** for telling no-stage from staged epochs (HB_1): flat share 0.79, saturation share 0.85, amplitude 0.39, `hf_ratio` 0.66, `mains_ratio` 0.74 (the two spectral ratios are now undefined for flat epochs).

### D18. `[x]` When one channel is unusable and the other is clean, does the headband still give a stage? (`one_channel.py`; asked 2026-10-27)

**Locked 2026-10-27** with the thresholds I proposed (a channel is bad at 10 times or 0.1 times the night's typical amplitude, or flat for at least half of the epoch), because no other values were asked for. Sensitivity run at 5 times and 0.2 times. Full text in `docs/definitions.md`, "One bad channel while the other is clean". It counts how often a clean channel was available. It does not stage from one channel.

**Results on all 128 nights** (`results/one_channel.json`). Main thresholds (bad at 10 times or 0.1 times the night's typical amplitude, or flat for at least half of the epoch):
- Of the 4,881 no-stage epochs, **1.9 % have both channels clean, 38.9 % have exactly one clean channel (1,897 epochs) and 59.2 % have both bad.** With the stricter thresholds (5 times, 0.2 times): 0.7 %, 37.7 % and 61.5 %.
- **The headband does stage most one-clean-channel epochs**: only 33.4 % of them have no stage (25.6 % with the stricter thresholds). Its agreement with the human on those epochs is 81.2 % (3,768 epochs), against 87.1 % when both channels are clean (110,361 epochs) and 65.8 % when both are bad (761 epochs).
- When exactly one channel is bad in a no-stage epoch, it is HB_1 (AF7) in 62.9 % of cases and HB_2 in 37.1 %.
- **Time a clean channel could recover:** 948 of the 2,440 no-stage minutes (39 %), but spread over few nights (median 0 minutes per night, IQR 0 to 0.6). Those 1,897 epochs are 1.6 % of all 120,095 epochs, so a perfect fallback could raise the share of staged epochs by at most about that much. Whether one channel alone would stage well was not tested.

### D19. `[x]` Does the headband signal carry what the human stages imply, do the two channels agree, and what are the amplitude spikes? (`stage_signature.py`; asked 2026-10-27)

**Locked 2026-10-27**, before any result was computed (full text in `docs/definitions.md`, "What the headband signal looks like for each human stage"). It checks consistency between the headband signal and the PSG-based human labels, not the labels themselves.

**Bug found and fixed while testing (2026-10-27).** A completely constant epoch at a non-zero level, such as the −2999.9 µV rail, does not have exactly zero variance in floating point. Its standard deviation came out around 1e-13 and its "spectrum" was rounding noise (which read as 100 % delta), so flat epochs were treated as live signal in the band-power features and in the earlier `hf_ratio` and `mains_ratio` features of D17. A *constant* epoch (every sample identical, `ptp == 0`, `constant_epochs` in the code) now has `std_uv` of exactly 0 and undefined spectral features. A test covers it. A *flat* epoch keeps its D17 meaning: both channels with `flat_share` of at least 0.5. Every constant epoch is flat, but a flat epoch can still contain some signal. The D17 conclusions about `flat_share` were not affected.

**Results on all 128 nights** (`results/stage_signature.json`, `figures/signals/fig17_stage_signature.png`; this replaces earlier 30-night numbers, which agreed in the main):
- **Stage signature.** Share of 0.5 to 30 Hz power in delta (0.5 to 4 Hz), median over nights, channels combined: N3 96.2 % (69 nights), N2 87.7 %, W 86.0 %, REM 83.9 %, N1 80.6 %. N3 also has the least of every other band. REM has the most theta (7.5 %). All stages are delta-dominated (1/f).
- **Separation within a night (median AUROC over nights).** N3 against N2 by delta: 0.86 over 62 nights (0.84 for each channel alone). N2 against N1: beta 0.20 (N2 has much less beta), delta 0.63. W against sleep: beta 0.72, amplitude 0.66. **REM against N2: beta 0.76**, theta 0.58, delta 0.40. With 30 nights, REM against N2 had looked hardly separable; with all nights it separates by beta (REM has more high-frequency activity), though not by delta or amplitude.
- **Reading.** The human's N3 labels line up clearly with the headband signal, and the N2, N1, W and REM labels line up to a moderate degree. This is consistency, not proof that the human labels are right: the human saw eye and chin channels that the headband does not have.
- **The two channels agree.** Median Spearman correlation between HB_1 and HB_2 per night: 0.85 for amplitude and 0.85 for delta share. Combining them gives about the same separation as either alone (0.86 against 0.84 each), so the value of the second channel is as a fallback, not for accuracy on a good night.
- **Amplitude spikes** (at least 3 times the night's median): 6.8 % of live epochs, 2.7 times over-represented in W (42.6 % of spikes against 16.0 % of epochs), 1.4 times in N3, and under-represented in N2 (0.6 times) and REM (0.7). Correlation with the accelerometer: median Spearman 0.23 over the 94 nights that have it.

### D20. `[x]` How well does the headband tell deep sleep from the rest, as a stimulation classifier would need? (`three_class.py`; asked 2026-10-27)

**Locked 2026-10-27**, before any result was computed (full text in `docs/definitions.md`, "Three-class view for stimulation gating"). Whether a stimulation classifier uses these three classes is an assumption, not something Bitbrain has stated in the data. The mapping from the five stages (REM counted as light) is our assumption, and a run without REM tests it.

**Results** (`results/three_class.json`; mapping W to wake, N1, N2 and REM to light, N3 to deep):
- **Three-class agreement is 93.0 %** (114,890 epochs), against 86.7 % for five stages. Wake: precision 86.3 %, recall 87.7 %.
- **Deep sleep is the weak point.** Recall 58.4 %, precision 69.1 %, F1 63.3 %. With a no-stage epoch counted as missed, recall is 53.8 %. 41.3 % of the human's deep epochs are called light and 0.3 % wake.
- **Deep calls that are wrong are almost never wakefulness.** Of the 4,068 epochs where the headband says deep, the human scored 6 as wake (0.15 %), 1,251 as light (30.8 %, N2 in 30.7 %) and 2,811 as deep (69.1 %). So the cost of a wrong deep call is mainly efficacy (stimulating N2), not safety.
- **Per night:** 61 nights have at least 20 human N3 epochs. The median share found is 0.59 (IQR 0.23 to 0.80), and in 9 of the 61 the headband finds none of it. Leaving out REM changes nothing material (accuracy 92.4 %, deep recall 58.4 %).
- **Caveat:** these are the dataset's headband labels, which may not come from the classifier that a stimulation system uses (and that one runs online, with past epochs only).

---

## C. Questions only Bitbrain can answer

Collect these for the email to Bitbrain. The data can't settle them.

### C1. `[ ]` How the consensus label was built (INV-1)

`README` says a fourth scorer broke three-way ties, but the sidecar says only "majority vote of 3". Which epochs went to the tie-breaker? Are the individual scorers' labels available? They would allow measuring inter-scorer uncertainty per epoch, which feeds directly into Phase 3.

### C2. `[ ]` Were the shipped AI labels predicted out-of-fold, and were the folds grouped by `pid`? (INV-6)

If not, AI agreement is optimistic. **Partly answered 2026-10-26:** the dataset README says the model was trained and validated with a cross-validation procedure, but not whether the folds kept all nights of one person together. **Until that is answered:** it is stated as a limitation in `REPORT.md`.

### C3. `[~]` What triggers the AI's `-2`? (INV-3)

**Still open:** the actual criterion, and why epochs with one wild channel and one clean one get the flag (D18).

Is it a threshold on the model's output or a separate artefact detector? Does it distinguish an artefact from a missing signal? Related: was the headband switched off or removed at the end of the long-tail nights (A3)?

### C4. `[ ]` Was sub-1 recorded after sleep onset? (A2)

### C7. `[ ]` Which model configuration produced the shipped BOAS labels? (section E)

The cited paper uses four other datasets and reports lower accuracy than the 87 % on BOAS. Were the BOAS labels made in the real-time (causal) or the offline configuration (this changes how to read the errors near stage changes, and whether the labels used future epochs)? Which model version, and how were the folds built (by person, given that some people have several nights)? The README says the model was trained and validated on BOAS with cross-validation.

### C8. `[ ]` What does a published "84 % with the headband" figure count?

A public presentation by Bitbrain (October 2026) reported 87 % with PSG and 84 % with the headband, for N=130. The dataset README gives 87.13 % and 86.71 % for 128 nights. Our figure for the headband is 86.74 % where both labels are a stage, and **83.21 % when a headband no-stage epoch counts as an error** (A1, D1). So the presentation's 84 % may be an end-to-end figure, which the README's would not be. Which epochs does each figure count? Is N=130 the same data as the 128 nights?

### C6. `[x]` Are the `sub-N` numbers in recording order? (D15)

**Resolved 2026-10-27 from the EDF headers (D17 Q3):** for all 20 people with more than one night, the EDF start dates are in the same order as the `sub-N` numbers. The note below is kept for the record.

The dataset has no acquisition dates in the label files. The analysis of first against later nights assumes that a person's nights are ordered by the number. Only Bitbrain, or the EDF headers (Phase 3), can confirm it.

### C5. `[ ]` Why is N3 low even in young adults? (A8)

The 18–29 median is 7.7 % N3 as a share of sleep, about half the textbook range. Was N3 scored conservatively, or does something about the recording setting or population explain it?

---

## D. Documentation and repo housekeeping

### H1. `[x]` Channel names in the inventory dropped the `PSG_` prefix (ARGON)

**Fixed 2026-10-24.** `docs/data_inventory.md` section 4 now lists the exact names (`PSG_F3` … `PSG_SPO2`).

### H2. `[~]` Cosmetic mismatches in the dataset metadata (INV-8, INV-9, INV-11, INV-12)

None of these affect Phase 0, which uses labels only.
- **PSG model name** (`README` vs sidecar): probably two names for different parts of the same Micromed system. Not checked.
- **Headband `EEGReference` "M1"**: **resolved 2026-10-27, the metadata is right.** The manufacturer's public documentation places the reference on the mastoids, so the earlier suspicion that it was copied from the PSG sidecar was wrong. The sidecar's `M1 (left mastoid)` is consistent with it.
- **`PSG_PULSE` units "unknown"**: unresolved, but irrelevant to Phase 0.
- **Esparza-Iaizzo citation year (2024 vs 2026)**: the DOI says 2026, so the "(2024)" in the reference list is probably a typo. Cite by DOI.

### H3. `[x]` The inventory's audit checks aren't reproducible from the repo (ARGON)

**Resolved 2026-10-24.** `src/boas_unc/audit_inventory.py` re-runs 44 checks covering inventory sections 2–8. Each check compares the files against the value the inventory documents (the `DOCUMENTED` dict in the script). It writes `results/inventory_audit.json` and exits non-zero on any mismatch.

Run it with `uv run python -m boas_unc.audit_inventory`. Result on 2026-10-24: 44/44 pass.

If the doc changes, update `DOCUMENTED` in the same edit.

### H4. `[x]` The package folder name doesn't match the project rules (CLAUDE)

**Resolved 2026-10-24.**
- `src/boas_uncertainty/` had already been deleted.
- `pyproject.toml` now builds the module `boas_unc` (`[tool.uv.build-backend] module-name`). The dead `boas-uncertainty = "boas_uncertainty:main"` script entry was removed.
- Added `src/boas_unc/__init__.py`. `import boas_unc` works after `uv sync`.
- Added `.gitignore` for `data/boas/`, `.venv/` and caches. `README.md` says the dataset clone is gitignored, but there was no `.gitignore`.

### H5. `[x]` There will be two loaders unless the inventory scripts move to `load.py` (ARGON)

**Resolved 2026-10-26.** `subjects`, `load_night`, `load_all` and `runs` now live in `src/boas_unc/load.py`, with tests in `tests/test_load.py`. `inventory_figures.py` and `audit_inventory.py` import them from there. After the change the audit still passes 44/44, and `results/inventory_counts.json` and `results/inventory_audit.json` are byte-identical to before.

### H6. `[x]` Figure fixes from Argon's second review (ARGON)

**Fixed 2026-10-24** in `inventory_figures.py`:
- Figure 1: the grey bands now have an edge line, so single-epoch `-2` runs (30 s, narrower than a pixel at this scale) are visible.
- Figure 2: boxes are now centred on their tick even when only one or two sources use a code (codes `8` and `-2`).
- Figure 2: the y-axis now says "% of all recorded epochs (not % of sleep)" (see D9).

---

## E. Notes from the Bitbrain paper (read 2026-10-27)

The paper cited in the dataset README is Esparza-Iaizzo et al., *Automatic sleep scoring for real-time monitoring and stimulation in individuals with and without sleep apnea*, Computers in Biology and Medicine 205 (2026) 111560, `doi:10.1016/j.compbiomed.2026.111560`. Open access (CC BY). The journal PDF was supplied by the user (`1-s2.0-S001048252600123X-main.pdf`, not tracked in git). The June 2024 bioRxiv preprint was read first, and where the two differ the journal numbers are used.

- **BOAS is not in the paper.** The data availability statement says the study used exclusively four open datasets: SleepEDF-78, ISRUC (patients with sleep apnea) and the two Dreem sets, DOD-H (healthy) and DOD-O (patients). Neither the BOAS data nor the figures in the BOAS README (87.13 % and 86.71 %) appear in it. So the README cites the paper for the model, and the BOAS results themselves are not documented there (C7).
- **Model:** a convolutional network (design from Youness 2020) on raw EEG resampled to 100 Hz. An epoch encoder (stacks of 1D convolutions, ending in a 64-unit dense layer) turns each 30 s epoch into a vector. A second part (the "Global" submodel) applies 1D convolutions over a sequence of epoch vectors. In real-time (causal) mode its kernels look only at past epochs, and in offline mode at past and future epochs. With kernel size 3 the receptive field is 7 epochs, 3.5 minutes.
- **Splits:** by subject, ten folds, 90 % and 10 %.
- **Reported numbers (journal version):** one frontal channel, real-time: 79.3 % on average over the datasets, 82.7 % in healthy people, 76.0 % in patients (a drop of 6.7 %). The best offline setting, which uses 3.5 minutes of future data, is 81.1 %. More channels or other sensors did not help on average. The preprint had 78.5 %, 81.4 % and 75.5 %. Wake, N2 and N3 are reported near 80 %, REM near 70 % and N1 lowest (preprint). Human inter-rater reliability is given as 80 to 85 %.
- **Aim:** real-time monitoring and closed-loop stimulation. The text points to slow-wave stimulation work in its references, but does not name N3 as the target stage.
- **Not in the paper:** sleep metrics (total sleep time, efficiency, latencies, WASO), per-night or per-person breakdowns, handling of missing-signal epochs, abstention or confidence analysis, and any wearable data. It cites uncertainty work by others (SleepTransformer) in the references only.
- **What this means for our findings:** the stage-level patterns (N1 hardest, N3 taken for N2) agree with what they report, so they are not new. The clinical-metric accounting, the dropout analysis, the distance-to-change breakdown and the abstention baselines are not in the paper. The agreement on BOAS (87 %) is higher than the paper's figures on public datasets (82.7 % in healthy people, real-time), and the BOAS README does not say whether the shipped labels come from the real-time or the offline configuration.
