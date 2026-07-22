"""Pre-registered health indicators (protocol section 4), computed from raw vibration.

Each HI maps one acquisition record (n_samples x n_channels) to a scalar. A
bearing's HI TRAJECTORY is the per-record HI over its life. HI names are frozen in
configs/grids.yaml; the primary HI is RMS of the horizontal channel (monotone,
standard). noisy_proxy() realizes the measurement-noise axis sigma (protocol s.4):
the analyst observes an imperfect summary of the true health state.
"""
import numpy as np


def rms(sig):
    return float(np.sqrt(np.mean(np.asarray(sig, float) ** 2)))


def kurtosis(sig):
    s = np.asarray(sig, float)
    s = s - s.mean()
    second_moment = np.mean(s ** 2)
    return float(np.mean(s ** 4) / (second_moment ** 2 + 1e-12))


def peak2peak(sig):
    s = np.asarray(sig, float)
    return float(s.max() - s.min())


def band_energy(sig, fs=25600.0, lo=1000.0, hi=5000.0):
    """Fraction of spectral energy in [lo, hi] Hz (defect frequencies live here)."""
    s = np.asarray(sig, float)
    n = len(s)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    P = np.abs(np.fft.rfft(s)) ** 2
    m = (f >= lo) & (f < hi)
    return float(P[m].sum() / (P.sum() + 1e-12))


HI_REGISTRY = {"rms": rms, "kurtosis": kurtosis, "p2p": peak2peak, "band_energy": band_energy}


def compute_hi(record, hi_name="rms", channel=0):
    """record: (n_samples, n_channels) array for one acquisition -> scalar HI."""
    rec = np.asarray(record, float)
    ch = rec[:, channel] if rec.ndim == 2 else rec
    return HI_REGISTRY[hi_name](ch)


def noisy_proxy(hi_value, sigma, rng):
    """Observed = true * exp(sigma * eps): the measurement-noise (sigma) axis."""
    hv = np.asarray(hi_value, float)
    return hv * np.exp(sigma * rng.standard_normal(hv.shape))
