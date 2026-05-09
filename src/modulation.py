"""
Digital modulation schemes for 5G NR.

Supports QPSK, 16QAM, and 64QAM with Gray coding,
following 3GPP TS 38.211 modulation mapper conventions.
"""
from __future__ import annotations

import numpy as np


def _qpsk_map() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=complex
    )


def _qam16_map() -> np.ndarray:
    levels = np.array([-3, -1, 3, 1])
    norm = 1 / np.sqrt(10)
    points = []
    for I in levels:
        for Q in levels:
            points.append(complex(I, Q))
    return norm * np.array(points, dtype=complex)


def _qam64_map() -> np.ndarray:
    levels = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
    norm = 1 / np.sqrt(42)
    points = []
    for I in levels:
        for Q in levels:
            points.append(complex(I, Q))
    return norm * np.array(points, dtype=complex)


CONSTELLATIONS = {
    "QPSK": (_qpsk_map(), 2),
    "16QAM": (_qam16_map(), 4),
    "64QAM": (_qam64_map(), 6),
}


def bits_per_symbol(scheme: str) -> int:
    if scheme not in CONSTELLATIONS:
        raise ValueError(f"Unknown modulation: {scheme}")
    return CONSTELLATIONS[scheme][1]


def modulate(bits: np.ndarray, scheme: str) -> np.ndarray:
    if scheme not in CONSTELLATIONS:
        raise ValueError(f"Unknown modulation: {scheme}")
    constellation, bps = CONSTELLATIONS[scheme]

    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(bits)) % bps
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    groups = bits.reshape(-1, bps)
    indices = groups.dot(1 << np.arange(bps - 1, -1, -1))
    return constellation[indices]


def demodulate(symbols: np.ndarray, scheme: str) -> np.ndarray:
    if scheme not in CONSTELLATIONS:
        raise ValueError(f"Unknown modulation: {scheme}")
    constellation, bps = CONSTELLATIONS[scheme]

    symbols = np.asarray(symbols, dtype=complex).ravel()
    dists = np.abs(symbols[:, None] - constellation[None, :]) ** 2
    indices = np.argmin(dists, axis=1)

    bits = ((indices[:, None] >> np.arange(bps - 1, -1, -1)) & 1).astype(np.uint8)
    return bits.ravel()