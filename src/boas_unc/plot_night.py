"""Plot a headband night to look at it: overview of the whole night, and zoomed raw traces.

Reads only a complete headband EDF (size and SHA256 checked) from the SD-card folder, read-only, and the
label files in data/boas. Writes PNGs into figures/signals/ (not the EDF).

  uv run python -m boas_unc.plot_night sub-1                    # overview + two automatic zoom windows
  uv run python -m boas_unc.plot_night sub-1 sub-3 --seconds 90 # several nights, longer zoom windows
  uv run python -m boas_unc.plot_night sub-1 --zoom-at 2.04     # zoom starting 2.04 h after the recording start
  uv run python -m boas_unc.plot_night --all                    # every complete night
  uv run python -m boas_unc.plot_night --file my_night.edf      # any EDF with channels HB_1 and HB_2, no labels needed

A BOAS night has labels, so its overview also shows both hypnograms. With --file there are no labels and no
checksum: the overview shows the EEG amplitude, flat epochs, the spectrogram and (if the file has them) movement.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from boas_unc import edf
from boas_unc import signal_quality as sq
from boas_unc import stage_signature as ss
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from boas_unc.inventory_figures import COLORS, GRID, HYP_LABELS, HYP_Y, INK, INK2, MUTED, SEQ_BLUE, SURFACE, style
from boas_unc.load import AI_MISSING, load_night, runs

OUT_DIR = sq.ROOT / "figures" / "signals"
BAND = (0.5, 35.0)   # Hz, band-pass for the zoomed traces (the usual range for looking at sleep EEG)
PAD_S = 10.0         # seconds of signal before and after a zoom window, so the filter edges are out of view
STD_FLOOR = 0.1      # microvolts; a flat epoch (std 0) is drawn here on the log axis
Y_LIMIT_UV = 150.0   # zoom y-axis limit (microvolts)
LEVEL_NOTE_UV = 1000.0  # a raw median level beyond this is printed on the zoom plot
SPEC_BAND = (0.5, 40.0)  # Hz shown in the spectrogram
SPEC_PERCENTILES = (2, 98)  # colour range of the spectrogram, so a few huge epochs do not wash it out


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float] = BAND, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    sos = sps.butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def window_trace(x: np.ndarray, fs: float, start_s: float, seconds: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Band-passed samples of [start, start + seconds), their times, and the raw median level of the window.

    The filter runs on a padded stretch so its edge effects are cut off.
    """
    lo = max(0, int(round((start_s - PAD_S) * fs)))
    a, b = int(round(start_s * fs)), int(round((start_s + seconds) * fs))
    hi = min(len(x), b + int(round(PAD_S * fs)))
    filtered = bandpass(x[lo:hi], fs)
    seg = filtered[a - lo:b - lo]
    return seg, np.arange(a, a + len(seg)) / fs, float(np.median(x[a:b]))


def epoch_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and the power spectral density (µV²/Hz) of every whole epoch, in the SPEC_BAND."""
    ep = edf.epochs(x, fs, sq.EPOCH_S)
    freqs, psd = sps.welch(ep, fs=fs, nperseg=int(sq.WELCH_S * fs), noverlap=0, axis=1)
    keep = (freqs >= SPEC_BAND[0]) & (freqs <= SPEC_BAND[1])
    return freqs[keep], psd[:, keep]


def to_db(psd: np.ndarray) -> np.ndarray:
    """Power in decibels; a zero (flat) epoch has no power and becomes NaN, not minus infinity."""
    with np.errstate(divide="ignore"):
        return np.where(psd > 0, 10 * np.log10(np.where(psd > 0, psd, 1.0)), np.nan)


def pick_windows_unlabelled(flat: np.ndarray, std: np.ndarray, epoch_s: float = sq.EPOCH_S) -> list[tuple[str, float]]:
    """Zoom windows for a file without labels: a typical live epoch, and the start of the first flat stretch.

    The typical epoch is the live (not flat) one whose amplitude is closest to the median of the live epochs.
    The flat stretch is the first run of at least LONG_RUN flat epochs, or else the first flat epoch.
    """
    out = []
    live = np.flatnonzero(~flat)
    if len(live):
        typical = live[np.argmin(np.abs(std[live] - np.median(std[live])))]
        out.append(("a typical epoch", float(typical) * epoch_s))
    gaps = runs(flat.astype(int), 1)
    first = ([g for g in gaps if g[1] >= sq.LONG_RUN] or gaps or [None])[0]
    if first is not None:
        out.append(("start of the first flat stretch", max(0, first[0] - 1) * epoch_s))
    return out


def pick_windows(hb_ai: np.ndarray, seconds: float, epoch_s: float = sq.EPOCH_S) -> list[tuple[str, float]]:
    """Automatic zoom windows (label, start in seconds): a quiet N2 stretch, and the start of the first dropout.

    The N2 window sits in the middle of the first run of at least LONG_RUN headband N2 epochs. The dropout window
    starts one epoch before the first run of at least LONG_RUN no-stage epochs, or before the first no-stage epoch.
    """
    out = []
    n2 = [r for r in runs(hb_ai, 2) if r[1] >= sq.LONG_RUN]
    if n2:
        start, length = n2[0]
        out.append(("headband N2", (start + length / 2) * epoch_s))
    gaps = runs(hb_ai, AI_MISSING)
    long_runs = [r for r in gaps if r[1] >= sq.LONG_RUN]
    first = (long_runs or gaps or [None])[0]
    if first is not None:
        out.append(("start of the first no-stage stretch", max(0, first[0] - 1) * epoch_s))
    return out


def stage_text(night, start_s: float, seconds: float) -> str:
    """Human and headband labels of the epochs that a window covers."""
    names = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 8: "8", AI_MISSING: "-2"}
    first, last = int(start_s // sq.EPOCH_S), int(np.ceil((start_s + seconds) / sq.EPOCH_S))
    rows = night.iloc[first:last]
    hum = " ".join(names.get(int(c), "?") for c in rows.psg_hum)
    hb = " ".join(names.get(int(c), "?") for c in rows.hb_ai)
    return f"epochs {first}-{last - 1}   human: {hum}   |   headband: {hb}"


def shade_special(ax, hours: np.ndarray, codes: np.ndarray, special: int, step: float):
    for start, length in runs(codes, special):
        ax.axvspan(hours[start], hours[start] + length * step, fc=MUTED, ec=MUTED, alpha=0.35, lw=0.6)


def hypnogram(ax, hours: np.ndarray, codes: np.ndarray, color: str, title: str):
    step = sq.EPOCH_S / 3600
    y = np.array([HYP_Y.get(c, np.nan) for c in codes], dtype=float)
    ax.step(np.append(hours, hours[-1] + step), np.append(y, y[-1]), where="post", color=color, lw=1.2)
    for code in (8, AI_MISSING):
        shade_special(ax, hours, codes, code, step)
    ax.set_yticks(range(5), HYP_LABELS)
    ax.set_ylim(-0.5, 4.5)
    ax.set_ylabel(title, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.5)


def draw_spectrogram(ax, freqs: np.ndarray, db: np.ndarray, n_epochs: int):
    """Spectrogram of one channel: hours on the x-axis, frequency on the y-axis, power as colour."""
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE).with_extremes(bad=to_rgba(MUTED, 0.25))
    finite = db[np.isfinite(db)]
    lo, hi = np.percentile(finite, SPEC_PERCENTILES) if len(finite) else (0.0, 1.0)
    ax.imshow(np.ma.masked_invalid(db).T, origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
              extent=[0, n_epochs * sq.EPOCH_S / 3600, freqs[0], freqs[-1]], interpolation="nearest")
    ax.set_ylabel(f"{sq.CHANNELS[0]} (Hz)", fontsize=9)


BAND_COLORS = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#104281"]  # an ordinal blue ramp, low to high frequency


def draw_bands(ax, bands, n_epochs: int):
    """Share of EEG power in each band over the night (stacked), from stage_signature.relative_band_powers."""
    step = sq.EPOCH_S / 3600
    hours = (np.arange(n_epochs) + 0.5) * step
    ax.stackplot(hours, [np.nan_to_num(bands[b].to_numpy()[:n_epochs]) for b in ss.BANDS], colors=BAND_COLORS,
                 labels=[f"{b} {lo:g}-{hi:g} Hz" for b, (lo, hi) in ss.BANDS.items()], linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0", "50", "100 %"])
    ax.set_ylabel("Share of\nEEG power", fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28), labelcolor=INK2)


def bands_of(path: Path, header: dict):
    """Per-epoch relative band powers of the channels combined (D19)."""
    return ss.night_views(path, header)[ss.COMBINED]


def fig_overview(night, joined, movement, header: dict, sub: str, out: Path, spectrum=None, bands=None):
    """Whole night: hypnograms (if there are labels), EEG amplitude per epoch, flat epochs, the share of power by
    band (bands = table from bands_of), the spectrogram of the first channel (spectrum = (frequencies, psd) from
    epoch_psd) and movement when available."""
    n = len(joined)
    step = sq.EPOCH_S / 3600
    hours = np.arange(n) * step
    layout = ([("hum", 1.0), ("hb", 1.0)] if night is not None else []) + [("std", 2.2), ("flat", 0.55)]
    layout += ([("bands", 1.5)] if bands is not None else []) + ([("spec", 1.6)] if spectrum is not None else [])
    layout += [("move", 1.1)] if movement is not None else []
    fig, axes = plt.subplots(len(layout), 1, figsize=(12, 1.3 * sum(h for _, h in layout) + 1.2), sharex=True,
                             gridspec_kw={"height_ratios": [h for _, h in layout]})
    ax = dict(zip([k for k, _ in layout], np.atleast_1d(axes)))
    if night is not None:
        hypnogram(ax["hum"], hours, night.psg_hum.to_numpy()[:n], COLORS["psg_hum"], "Human (PSG)")
        hypnogram(ax["hb"], hours, night.hb_ai.to_numpy()[:n], COLORS["hb_ai"], "Headband AI")
    for ch, color in zip(sq.CHANNELS, (COLORS["hb_ai"], INK2)):
        ax["std"].plot(hours, np.maximum(joined[f"{ch}_std_uv"], STD_FLOOR), color=color, lw=0.8, label=ch, alpha=0.9)
    ax["std"].set_yscale("log")
    ax["std"].set_ylabel("EEG amplitude: std\nof each 30 s epoch (µV)", fontsize=9)
    ax["std"].grid(axis="y", color=GRID, lw=0.5)
    ax["std"].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    flat = joined.flat_epoch.to_numpy()
    ax["flat"].bar(hours[flat], 1.0, width=step, align="edge", color="#d03b3b")
    ax["flat"].set_yticks([])
    ax["flat"].set_ylabel("Flat\nepoch", fontsize=9)
    if bands is not None:
        draw_bands(ax["bands"], bands, n)
    if spectrum is not None:
        draw_spectrogram(ax["spec"], spectrum[0], to_db(spectrum[1][:n]), min(n, len(spectrum[1])))
    if movement is not None:
        ax["move"].plot(hours[:len(movement)], movement[:n], color=INK2, lw=0.8)
        ax["move"].set_ylabel("Movement\n(accel. std, g)", fontsize=9)
        ax["move"].grid(axis="y", color=GRID, lw=0.5)
    list(ax.values())[-1].set_xlabel("Hours since recording start")
    list(ax.values())[-1].set_xlim(0, hours[-1] + step)
    flat_n = int(flat.sum())
    labels = (f", {int(joined.missing.sum())} with no headband stage ({100 * joined.missing.mean():.1f} %)"
              if night is not None else "")
    fig.suptitle(f"{sub}  |  recorded {header['start']:%Y-%m-%d %H:%M}  |  {n} epochs{labels}, {flat_n} flat",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    note = ("Amplitude is a one-number summary of each epoch (log scale; a flat epoch is drawn at the floor): the raw EEG is "
            "in the zoom plot. Red bars: both channels flat for at least half of the epoch.")
    if bands is not None:
        note += " Power shares: the channels' mean; deep sleep (N3) is mostly delta."
    if spectrum is not None:
        note += " Spectrogram: darker = more power; grey = a flat epoch."
    if night is not None:
        note = "Grey bands: no stage (human code 8, headband -2). " + note
    fig.text(0.01, 0.005, textwrap.fill(note, 165), color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_zoom(night, path: Path, header: dict, sub: str, windows: list[tuple[str, float]], seconds: float, out: Path):
    """Band-passed raw traces of both channels (and the accelerometer norm, if present) for each window."""
    has_imu = all(a in [s["label"] for s in header["signals"]] for a in sq.ACCEL)
    n_rows = len(sq.CHANNELS) + 1 + (1 if has_imu else 0)  # filtered channels, raw HB_1, accelerometer
    fig, axes = plt.subplots(n_rows, len(windows), figsize=(7 * len(windows), 2.0 * n_rows + 1.6), squeeze=False)
    for col, (label, start_s) in enumerate(windows):
        for row, ch in enumerate(sq.CHANNELS):
            sig = header["signals"][edf.signal_index(header, ch)]
            x = edf.read_signal(path, ch, header)
            seg, t, level = window_trace(x, sig["sampling_hz"], start_s, seconds)
            ax = axes[row, col]
            ax.plot(t - start_s, seg, color=COLORS["hb_ai"] if row == 0 else INK2, lw=0.6)
            ax.set_ylim(-Y_LIMIT_UV, Y_LIMIT_UV)
            ax.grid(color=GRID, lw=0.5)
            ax.set_ylabel(f"{ch} (µV)", fontsize=9)
            if abs(level) > LEVEL_NOTE_UV:
                ax.text(0.99, 0.92, f"raw level {level:.0f} µV: signal at the rail", transform=ax.transAxes, ha="right",
                        va="top", fontsize=8, color="#d03b3b")
        sig = header["signals"][edf.signal_index(header, sq.CHANNELS[0])]
        fs = sig["sampling_hz"]
        raw = edf.read_signal(path, sq.CHANNELS[0], header)[int(start_s * fs):int((start_s + seconds) * fs)]
        ax = axes[len(sq.CHANNELS), col]
        ax.plot(np.arange(len(raw)) / fs, raw, color=INK, lw=0.6)
        ax.set_ylabel(f"{sq.CHANNELS[0]} raw,\nunfiltered (µV)", fontsize=9)
        ax.grid(color=GRID, lw=0.5)
        if has_imu:
            norm = np.sqrt(sum(edf.read_signal(path, a, header).astype(float) ** 2 for a in sq.ACCEL))
            fs = header["signals"][edf.signal_index(header, sq.ACCEL[0])]["sampling_hz"]
            a, b = int(start_s * fs), int((start_s + seconds) * fs)
            ax = axes[-1, col]
            ax.plot(np.arange(b - a) / fs, norm[a:b], color=MUTED, lw=0.7)
            ax.set_ylabel("|accel| (g)", fontsize=9)
            ax.grid(color=GRID, lw=0.5)
        axes[0, col].set_title(f"{label}, from {start_s / 3600:.2f} h", fontsize=10)
        axes[-1, col].set_xlabel("Seconds into the window")
        if night is not None:
            fig.text(0.01 + col * 0.5, 0.005, stage_text(night, start_s, seconds)[:110], fontsize=7, color=MUTED)
    fig.suptitle(f"{sub}: headband EEG. Top rows band-passed {BAND[0]}-{BAND[1]} Hz (the filter rings after a step), "
                 "then the unfiltered signal", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def spectrum_of(path: Path, header: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch spectra of the first EEG channel."""
    sig = header["signals"][edf.signal_index(header, sq.CHANNELS[0])]
    return epoch_psd(edf.read_signal(path, sq.CHANNELS[0], header), sig["sampling_hz"])


def plot_file(path: Path, out_dir: Path, seconds: float, zoom_at_h: float | None) -> list[Path]:
    """Overview and zoom of any EDF with channels HB_1 and HB_2. No labels, no checksum (use for your own recordings).

    A file that is shorter than its header says is refused by the reader.
    """
    header = edf.read_header(path)
    labels = [s["label"] for s in header["signals"]]
    missing = [c for c in sq.CHANNELS if c not in labels]
    if missing:
        raise SystemExit(f"{path.name}: no channel named {missing}; the file has {labels}")
    quality = sq.night_quality(path, header)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview, zoom = out_dir / f"{path.stem}_overview.png", out_dir / f"{path.stem}_zoom.png"
    fig_overview(None, quality, sq.movement_per_epoch(path, header), header, path.stem, overview, spectrum_of(path, header),
                 bands_of(path, header))
    windows = ([("chosen window", zoom_at_h * 3600)] if zoom_at_h is not None else
               pick_windows_unlabelled(quality.flat_epoch.to_numpy(), quality[f"{sq.CHANNELS[0]}_std_uv"].to_numpy()))
    if windows:
        fig_zoom(None, path, header, path.stem, windows, seconds, zoom)
        return [overview, zoom]
    return [overview]


def plot_night(sub: str, edf_dir: Path, out_dir: Path, seconds: float, zoom_at_h: float | None) -> list[Path]:
    path = sq.edf_path(edf_dir, sub)
    if not sq.is_complete(path, sq.annex_key(sq.DATA, sub)):
        raise SystemExit(f"{sub}: the EDF is missing or not fully downloaded (size or SHA256 does not match)")
    header = edf.read_header(path)
    night = load_night(sq.DATA, sub).sort_values("epoch").reset_index(drop=True)
    joined = sq.align(night, sq.night_quality(path, header))
    out_dir.mkdir(parents=True, exist_ok=True)
    overview, zoom = out_dir / f"{sub}_overview.png", out_dir / f"{sub}_zoom.png"
    fig_overview(night, joined, sq.movement_per_epoch(path, header), header, sub, overview, spectrum_of(path, header),
                 bands_of(path, header))
    windows = [("chosen window", zoom_at_h * 3600)] if zoom_at_h is not None else pick_windows(night.hb_ai.to_numpy(), seconds)
    if windows:
        fig_zoom(night, path, header, sub, windows, seconds, zoom)
        return [overview, zoom]
    return [overview]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subs", nargs="*", help="recordings, for example sub-1 sub-3")
    parser.add_argument("--file", type=Path, default=None, help="plot one EDF file with channels HB_1 and HB_2, no labels")
    parser.add_argument("--all", action="store_true", help="every complete night")
    parser.add_argument("--seconds", type=float, default=60.0, help="length of the zoom windows (default 60)")
    parser.add_argument("--zoom-at", type=float, default=None, help="start of one zoom window, in hours since the start")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder (default figures/signals)")
    parser.add_argument("--edf-dir", type=Path, default=sq.EDF_DIR, help="folder with the headband EDFs")
    args = parser.parse_args(argv)
    style()
    if args.file is not None:
        if args.subs or args.all:
            parser.error("--file cannot be combined with recordings or --all")
        for p in plot_file(args.file, args.out, args.seconds, args.zoom_at):
            print(f"wrote {p}")
        return
    subs = sq.complete_nights(sq.DATA, args.edf_dir)[0] if args.all else args.subs
    if not subs:
        parser.error("give at least one recording, or --all")
    for sub in subs:
        for p in plot_night(sub, args.edf_dir, args.out, args.seconds, args.zoom_at):
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
