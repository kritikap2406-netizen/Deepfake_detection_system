"""
TrueSight — rPPG (remote photoplethysmography).

Every heartbeat pushes a small pulse of blood through facial skin
capillaries, causing a subtle, periodic change in skin color — too small
to see by eye, but detectable by averaging pixel color over a skin ROI
across many frames and looking for a periodic signal in the plausible
heart-rate frequency band (roughly 42-240 BPM -> 0.7-4.0 Hz).

A real video of a real face should show a detectable, sufficiently
periodic signal in this band. A generated/synthetic face often has no
underlying blood-flow model at all, so the "skin color" has no genuine
periodicity — the rPPG signal is closer to noise. That's the core signal
we're extracting: not the heart rate itself, but how strongly periodic
the extracted signal is within a physiologically plausible band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, welch

from configs_loader import load_config

_config = load_config()
RPPG_WINDOW_SECONDS = _config["physiological"]["rppg_window_seconds"]
MIN_FRAMES_REQUIRED = _config["physiological"]["min_frames_required"]

HEART_RATE_BAND_HZ = (0.7, 4.0)  # ~42-240 BPM, generous physiological range


@dataclass
class RppgAnalysis:
    dominant_freq_hz: float | None
    estimated_bpm: float | None
    periodicity_strength: float | None  # 0-1, higher = clearer pulse-like periodicity
    rppg_score: float                   # 0 (plausible/real-like) - 1 (implausible/fake-like)
    abstained: bool


def extract_mean_color_signal(cheek_crops: list[np.ndarray]) -> np.ndarray:
    """cheek_crops: list of BGR ROI patches, one per frame.
    Returns the green-channel mean per frame — green is the standard
    single-channel choice for rPPG since hemoglobin absorbs green light
    most strongly, giving the best signal-to-noise ratio of the three
    raw channels without needing a more complex chrominance combination.
    """
    if not cheek_crops:
        return np.array([])
    return np.array([crop[:, :, 1].mean() for crop in cheek_crops])  # BGR -> index 1 is green


def bandpass_filter(signal: np.ndarray, fps: float, band: tuple[float, float] = HEART_RATE_BAND_HZ) -> np.ndarray:
    """Butterworth bandpass, isolating the plausible heart-rate frequency
    range and removing slow lighting drift + high-frequency camera noise.
    """
    nyquist = fps / 2.0
    low, high = band[0] / nyquist, band[1] / nyquist
    low, high = max(1e-4, low), min(0.999, high)

    if low >= high:
        return signal - signal.mean()  # degenerate case (very low fps), skip filtering

    b, a = butter(N=3, Wn=[low, high], btype="band")
    detrended = signal - signal.mean()

    # filtfilt needs signal longer than ~3x filter order; guard short sequences
    if len(detrended) <= 3 * (max(len(a), len(b)) - 1):
        return detrended

    return filtfilt(b, a, detrended)


def analyze_rppg(cheek_crops: list[np.ndarray], fps: float) -> RppgAnalysis:
    n_frames = len(cheek_crops)

    if n_frames < MIN_FRAMES_REQUIRED:
        return RppgAnalysis(
            dominant_freq_hz=None, estimated_bpm=None,
            periodicity_strength=None, rppg_score=0.5, abstained=True,
        )

    raw_signal = extract_mean_color_signal(cheek_crops)
    filtered = bandpass_filter(raw_signal, fps)

    # Welch's method: robust power spectral density estimate, better
    # suited to short, noisy signals than a raw FFT periodogram.
    nperseg = min(len(filtered), max(8, int(fps * 2)))
    freqs, power = welch(filtered, fs=fps, nperseg=nperseg)

    band_mask = (freqs >= HEART_RATE_BAND_HZ[0]) & (freqs <= HEART_RATE_BAND_HZ[1])
    if not band_mask.any() or power[band_mask].sum() < 1e-12:
        return RppgAnalysis(
            dominant_freq_hz=None, estimated_bpm=None,
            periodicity_strength=0.0, rppg_score=0.7, abstained=False,
        )

    band_freqs = freqs[band_mask]
    band_power = power[band_mask]

    peak_idx = int(np.argmax(band_power))
    dominant_freq = float(band_freqs[peak_idx])
    estimated_bpm = dominant_freq * 60.0

    # Periodicity strength: how much of the in-band power is concentrated
    # at the peak vs spread across the band. A true pulse signal shows a
    # sharp peak; noise shows flat, spread-out power.
    periodicity_strength = float(band_power[peak_idx] / band_power.sum())

    rppg_score = round(1.0 - min(1.0, periodicity_strength * 3.0), 4)  # scale: sharp peak -> low (real-like) score

    return RppgAnalysis(
        dominant_freq_hz=round(dominant_freq, 3),
        estimated_bpm=round(estimated_bpm, 1),
        periodicity_strength=round(periodicity_strength, 4),
        rppg_score=rppg_score,
        abstained=False,
    )
