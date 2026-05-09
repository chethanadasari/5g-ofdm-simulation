"""
5G NR-style OFDM transceiver simulation.
Reference: 3GPP TS 38.211 (Physical channels and modulation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from modulation import bits_per_symbol, demodulate, modulate
from channel import apply_multipath, awgn, tdl_a_taps


@dataclass
class OFDMConfig:
    n_subcarriers: int = 64
    n_data_subcarriers: int = 52
    cp_length: int = 16
    modulation: str = "QPSK"
    subcarrier_spacing_hz: float = 15e3
    pilot_subcarriers: tuple = field(default_factory=lambda: (-21, -7, 7, 21))

    @property
    def sample_rate_hz(self) -> float:
        return self.n_subcarriers * self.subcarrier_spacing_hz

    @property
    def symbol_duration_s(self) -> float:
        return (self.n_subcarriers + self.cp_length) / self.sample_rate_hz

    def data_indices(self) -> np.ndarray:
        n = self.n_subcarriers
        center = n // 2
        guard = (n - self.n_data_subcarriers - 1) // 2
        all_idx = np.arange(guard, n - guard)
        all_idx = all_idx[all_idx != center]
        pilot_idx = np.array([center + p for p in self.pilot_subcarriers])
        return np.array([i for i in all_idx if i not in pilot_idx])


def ofdm_modulate(symbols: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    data_idx = cfg.data_indices()
    n_per_symbol = len(data_idx)

    pad = (-len(symbols)) % n_per_symbol
    if pad:
        symbols = np.concatenate([symbols, np.zeros(pad, dtype=complex)])
    n_ofdm_symbols = len(symbols) // n_per_symbol

    out = []
    for k in range(n_ofdm_symbols):
        block = symbols[k * n_per_symbol:(k + 1) * n_per_symbol]
        grid = np.zeros(cfg.n_subcarriers, dtype=complex)
        grid[data_idx] = block
        center = cfg.n_subcarriers // 2
        for p in cfg.pilot_subcarriers:
            grid[center + p] = 1.0 + 0j
        time = np.fft.ifft(np.fft.ifftshift(grid)) * np.sqrt(cfg.n_subcarriers)
        cp = time[-cfg.cp_length:]
        out.append(np.concatenate([cp, time]))
    return np.concatenate(out)


def ofdm_demodulate(rx: np.ndarray, cfg: OFDMConfig, channel_freq: np.ndarray | None = None) -> np.ndarray:
    sym_len = cfg.n_subcarriers + cfg.cp_length
    n_symbols = len(rx) // sym_len
    rx = rx[: n_symbols * sym_len]
    blocks = rx.reshape(n_symbols, sym_len)

    data_idx = cfg.data_indices()
    out = []
    for blk in blocks:
        time = blk[cfg.cp_length:]
        freq = np.fft.fftshift(np.fft.fft(time)) / np.sqrt(cfg.n_subcarriers)
        if channel_freq is not None:
            freq = freq / channel_freq
        out.append(freq[data_idx])
    return np.concatenate(out)


def estimate_channel_freq(cfg: OFDMConfig, taps: np.ndarray) -> np.ndarray:
    H = np.fft.fft(taps, cfg.n_subcarriers)
    return np.fft.fftshift(H)


def transmit(bits: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    syms = modulate(bits, cfg.modulation)
    return ofdm_modulate(syms, cfg)


def receive(rx: np.ndarray, cfg: OFDMConfig, n_bits: int,
            channel_freq: np.ndarray | None = None) -> np.ndarray:
    eq_syms = ofdm_demodulate(rx, cfg, channel_freq=channel_freq)
    bits = demodulate(eq_syms, cfg.modulation)
    return bits[:n_bits]


def ber_snr_sweep(cfg: OFDMConfig, snr_db_range, n_bits: int = 20000,
                  channel: str = "awgn", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bps = bits_per_symbol(cfg.modulation)
    syms_per = len(cfg.data_indices())
    bits_per_ofdm = syms_per * bps
    n_bits = int(np.ceil(n_bits / bits_per_ofdm)) * bits_per_ofdm

    bers = np.empty(len(snr_db_range))
    for i, snr_db in enumerate(snr_db_range):
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        tx = transmit(bits, cfg)

        if channel == "tdl-a":
            taps = tdl_a_taps(cfg.sample_rate_hz, rng=rng)
            rx = apply_multipath(tx, taps)
            H = estimate_channel_freq(cfg, taps)
        else:
            rx = tx
            H = None

        rx = awgn(rx, snr_db, rng=rng)
        rx_bits = receive(rx, cfg, n_bits, channel_freq=H)
        bers[i] = np.mean(bits != rx_bits)
    return bers


if __name__ == "__main__":
    cfg = OFDMConfig(modulation="QPSK")
    snrs = np.arange(0, 22, 2)
    print(f"OFDM Config: {cfg.n_subcarriers}-pt FFT, CP={cfg.cp_length}, "
          f"{cfg.modulation}, fs={cfg.sample_rate_hz/1e6:.2f} MHz\n")
    print("AWGN channel:")
    for snr, ber in zip(snrs, ber_snr_sweep(cfg, snrs, n_bits=20000)):
        print(f"  SNR = {snr:5.1f} dB   BER = {ber:.5f}")