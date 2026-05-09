"""
Wireless channel models: AWGN and tapped-delay-line multipath.

Multipath profile based on 3GPP TR 38.901 TDL-A (low delay spread)
for sub-6 GHz 5G NR scenarios.
"""
from __future__ import annotations

import numpy as np


def awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex white Gaussian noise at the given SNR (per-sample)."""
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_lin
    noise = (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    noise *= np.sqrt(noise_power / 2)
    return signal + noise


# 3GPP TDL-A normalised power profile (dB) and delay (ns).
TDL_A_DELAYS_NS = np.array([0.0, 30.5, 85.0, 145.0, 230.0])
TDL_A_POWERS_DB = np.array([0.0, -2.2, -4.0, -7.5, -12.0])


def tdl_a_taps(sample_rate_hz: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Build complex Rayleigh-faded TDL-A channel taps for the given sample rate."""
    rng = rng or np.random.default_rng()
    delays_s = TDL_A_DELAYS_NS * 1e-9
    sample_idx = np.round(delays_s * sample_rate_hz).astype(int)
    n_taps = sample_idx.max() + 1

    taps = np.zeros(n_taps, dtype=complex)
    powers = 10 ** (TDL_A_POWERS_DB / 10)
    powers /= powers.sum()

    for idx, p in zip(sample_idx, powers):
        h = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        taps[idx] += h * np.sqrt(p)
    return taps


def apply_multipath(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve signal with the channel impulse response."""
    return np.convolve(signal, taps, mode="full")[: len(signal)]