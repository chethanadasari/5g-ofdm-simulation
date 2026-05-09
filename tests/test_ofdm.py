"""Unit tests for the 5G OFDM transceiver."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modulation import (
    bits_per_symbol,
    demodulate,
    modulate,
    CONSTELLATIONS,
)
from channel import apply_multipath, awgn, tdl_a_taps
from ofdm_transceiver import (
    OFDMConfig,
    ber_snr_sweep,
    estimate_channel_freq,
    ofdm_demodulate,
    ofdm_modulate,
    receive,
    transmit,
)


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme,bps", [("QPSK", 2), ("16QAM", 4), ("64QAM", 6)])
def test_bits_per_symbol(scheme, bps):
    assert bits_per_symbol(scheme) == bps


@pytest.mark.parametrize("scheme", ["QPSK", "16QAM", "64QAM"])
def test_constellation_unit_power(scheme):
    constellation, _ = CONSTELLATIONS[scheme]
    assert np.isclose(np.mean(np.abs(constellation) ** 2), 1.0, atol=1e-9)


@pytest.mark.parametrize("scheme", ["QPSK", "16QAM", "64QAM"])
def test_modulate_demodulate_noiseless(scheme):
    rng = np.random.default_rng(0)
    bps = bits_per_symbol(scheme)
    bits = rng.integers(0, 2, size=bps * 1000, dtype=np.uint8)
    syms = modulate(bits, scheme)
    rx_bits = demodulate(syms, scheme)
    assert np.array_equal(bits, rx_bits)


def test_unknown_modulation_raises():
    with pytest.raises(ValueError):
        modulate(np.array([0, 1]), "256QAM")


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

def test_awgn_snr_close_to_target():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000)
    rx = awgn(sig, snr_db=10, rng=rng)
    noise = rx - sig
    sig_p = np.mean(np.abs(sig) ** 2)
    noise_p = np.mean(np.abs(noise) ** 2)
    measured_snr = 10 * np.log10(sig_p / noise_p)
    assert abs(measured_snr - 10) < 0.5


def test_tdl_a_taps_unit_power():
    rng = np.random.default_rng(0)
    powers = []
    for _ in range(500):
        taps = tdl_a_taps(15.36e6, rng=rng)
        powers.append(np.sum(np.abs(taps) ** 2))
    assert abs(np.mean(powers) - 1.0) < 0.1


def test_apply_multipath_length_preserved():
    sig = np.ones(100, dtype=complex)
    taps = np.array([1.0, 0.5, 0.2], dtype=complex)
    out = apply_multipath(sig, taps)
    assert len(out) == len(sig)


# ---------------------------------------------------------------------------
# OFDM TX / RX
# ---------------------------------------------------------------------------

def test_ofdm_modulate_length():
    cfg = OFDMConfig()
    syms = np.ones(len(cfg.data_indices()) * 5, dtype=complex)
    out = ofdm_modulate(syms, cfg)
    assert len(out) == 5 * (cfg.n_subcarriers + cfg.cp_length)


def test_ofdm_round_trip_noiseless():
    cfg = OFDMConfig(modulation="QPSK")
    rng = np.random.default_rng(0)
    n_per = len(cfg.data_indices())
    syms = (rng.choice([-1, 1], size=n_per * 4)
            + 1j * rng.choice([-1, 1], size=n_per * 4)) / np.sqrt(2)
    tx = ofdm_modulate(syms, cfg)
    rx = ofdm_demodulate(tx, cfg)
    assert np.allclose(rx, syms, atol=1e-9)


@pytest.mark.parametrize("scheme", ["QPSK", "16QAM", "64QAM"])
def test_full_chain_noiseless(scheme):
    cfg = OFDMConfig(modulation=scheme)
    rng = np.random.default_rng(0)
    bps = bits_per_symbol(scheme)
    n_bits = bps * len(cfg.data_indices()) * 4
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    tx = transmit(bits, cfg)
    rx_bits = receive(tx, cfg, n_bits)
    assert np.array_equal(bits, rx_bits)


def test_estimate_channel_freq_shape():
    cfg = OFDMConfig()
    taps = np.array([1.0, 0.5, 0.2], dtype=complex)
    H = estimate_channel_freq(cfg, taps)
    assert H.shape == (cfg.n_subcarriers,)


# ---------------------------------------------------------------------------
# BER sanity checks
# ---------------------------------------------------------------------------

def test_ber_decreases_with_snr_awgn():
    cfg = OFDMConfig(modulation="QPSK")
    snrs = np.array([0, 5, 10, 15])
    bers = ber_snr_sweep(cfg, snrs, n_bits=10_000, channel="awgn", seed=0)
    assert all(bers[i] >= bers[i + 1] - 0.01 for i in range(len(bers) - 1))
    assert bers[-1] < bers[0]


def test_ber_qpsk_high_snr_low():
    cfg = OFDMConfig(modulation="QPSK")
    bers = ber_snr_sweep(cfg, [20], n_bits=20_000, channel="awgn", seed=0)
    assert bers[0] < 1e-3


def test_ber_64qam_higher_than_qpsk_at_same_snr():
    snr = [10]
    qpsk = ber_snr_sweep(OFDMConfig(modulation="QPSK"), snr, n_bits=20_000, seed=0)
    qam64 = ber_snr_sweep(OFDMConfig(modulation="64QAM"), snr, n_bits=20_000, seed=0)
    assert qam64[0] > qpsk[0]


def test_multipath_recovers_with_zf():
    cfg = OFDMConfig(modulation="QPSK")
    bers = ber_snr_sweep(cfg, [25], n_bits=10_000, channel="tdl-a", seed=0)
    assert bers[0] < 0.05
