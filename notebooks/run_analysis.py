"""
Reproducible analysis script: BER vs SNR sweep + constellation diagrams.
Generates the plots used in the README. Run with:

    python notebooks/run_analysis.py

Outputs go to results/.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "src"))

from modulation import modulate
from ofdm_transceiver import (  # noqa: E402
    OFDMConfig,
    ber_snr_sweep,
    estimate_channel_freq,
    ofdm_demodulate,
    ofdm_modulate,
)
from channel import apply_multipath, awgn, tdl_a_taps  # noqa: E402

OUT = os.path.join(THIS, "..", "results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


# ---------------------------------------------------------------------------
# 1. BER vs SNR for QPSK / 16QAM / 64QAM (AWGN)
# ---------------------------------------------------------------------------

def plot_ber_awgn():
    snrs = np.arange(0, 26, 2)
    plt.figure(figsize=(7, 5))
    for scheme, marker in [("QPSK", "o"), ("16QAM", "s"), ("64QAM", "^")]:
        cfg = OFDMConfig(modulation=scheme)
        bers = ber_snr_sweep(cfg, snrs, n_bits=40_000, channel="awgn", seed=0)
        plt.semilogy(snrs, np.maximum(bers, 1e-6),
                     marker=marker, label=scheme, linewidth=1.5)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("OFDM BER vs SNR — AWGN channel")
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ber_awgn.png"))
    plt.close()
    print("  saved results/ber_awgn.png")


# ---------------------------------------------------------------------------
# 2. BER vs SNR — AWGN vs TDL-A multipath (QPSK)
# ---------------------------------------------------------------------------

def plot_ber_multipath():
    snrs = np.arange(0, 31, 2)
    cfg = OFDMConfig(modulation="QPSK")
    awgn_b = ber_snr_sweep(cfg, snrs, n_bits=40_000, channel="awgn", seed=1)
    tdla_b = ber_snr_sweep(cfg, snrs, n_bits=40_000, channel="tdl-a", seed=1)
    plt.figure(figsize=(7, 5))
    plt.semilogy(snrs, np.maximum(awgn_b, 1e-6), "o-", label="AWGN")
    plt.semilogy(snrs, np.maximum(tdla_b, 1e-6), "s-", label="TDL-A multipath + ZF eq.")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("QPSK OFDM — AWGN vs frequency-selective multipath")
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ber_multipath.png"))
    plt.close()
    print("  saved results/ber_multipath.png")


# ---------------------------------------------------------------------------
# 3. Received constellation at SNR = 15 dB (QPSK / 16QAM / 64QAM)
# ---------------------------------------------------------------------------

def plot_constellations(snr_db: float = 15.0):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    rng = np.random.default_rng(0)
    for ax, scheme in zip(axes, ["QPSK", "16QAM", "64QAM"]):
        cfg = OFDMConfig(modulation=scheme)
        n_per = len(cfg.data_indices())
        n_bits = 12 * n_per * {"QPSK": 2, "16QAM": 4, "64QAM": 6}[scheme]
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        syms = modulate(bits, scheme)
        tx = ofdm_modulate(syms, cfg)
        rx = awgn(tx, snr_db, rng=rng)
        eq = ofdm_demodulate(rx, cfg)
        ax.scatter(eq.real, eq.imag, s=8, alpha=0.4)
        ax.set_title(f"{scheme}  @  SNR = {snr_db:.0f} dB")
        ax.axhline(0, color="k", linewidth=0.4); ax.axvline(0, color="k", linewidth=0.4)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("In-phase"); ax.set_ylabel("Quadrature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "constellations.png"))
    plt.close()
    print("  saved results/constellations.png")


# ---------------------------------------------------------------------------
# 4. Channel frequency response example
# ---------------------------------------------------------------------------

def plot_channel_response():
    cfg = OFDMConfig()
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(3):
        taps = tdl_a_taps(cfg.sample_rate_hz, rng=rng)
        H = estimate_channel_freq(cfg, taps)
        f_axis = np.arange(-cfg.n_subcarriers // 2, cfg.n_subcarriers // 2) \
            * cfg.subcarrier_spacing_hz / 1e3
        ax.plot(f_axis, 20 * np.log10(np.abs(H) + 1e-9),
                label=f"realisation {i + 1}")
    ax.set_xlabel("Subcarrier frequency offset (kHz)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title("TDL-A channel frequency response (3 random realisations)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "channel_response.png"))
    plt.close()
    print("  saved results/channel_response.png")


if __name__ == "__main__":
    print("Generating analysis plots...")
    plot_ber_awgn()
    plot_ber_multipath()
    plot_constellations()
    plot_channel_response()
    print("Done. See the results/ folder.")