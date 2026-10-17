"""Simulated radio KPI time series and windowed next-window degradation labels.

Three channels per step: SINR (dB), BLER (0..1), HARQ retransmission count
(0..3). Each session is an AR(1) SINR trace with occasional fade/interference
episodes; BLER and HARQ retransmissions are derived from SINR with added
noise, which is how a real link's KPIs actually correlate. A window of
WINDOW_LEN consecutive steps is labeled 1 ("next-window degradation") when
the WINDOW_LEN steps immediately after it have mean SINR below a degradation
threshold or mean BLER above a degradation threshold; label 0 otherwise.
Everything here is synthetic and seeded; it is not measured radio data.
"""
import numpy as np

WINDOW_LEN = 64
STEPS_PER_SESSION = 320
SEED = 20260914

SINR_DEGRADED_DB = 2.0
BLER_DEGRADED = 0.25


def _simulate_session(rng):
    t = STEPS_PER_SESSION
    sinr = np.empty(t, dtype=np.float64)
    base = rng.uniform(8.0, 18.0)
    sinr[0] = base
    phi = 0.92
    for i in range(1, t):
        sinr[i] = base + phi * (sinr[i - 1] - base) + rng.normal(0.0, 0.5)

    # Occasional fade/interference episodes: sustained SINR dips.
    n_episodes = rng.poisson(2.2)
    for _ in range(n_episodes):
        start = rng.integers(0, t - 10)
        length = rng.integers(6, 30)
        depth = rng.uniform(6.0, 16.0)
        end = min(t, start + length)
        ramp = np.linspace(0.0, 1.0, end - start)
        dip = depth * np.sin(np.pi * ramp)
        sinr[start:end] -= dip

    # BLER is a noisy decreasing sigmoid of SINR (lower SINR -> higher BLER).
    bler_mid, bler_slope = 6.0, 0.55
    bler_clean = 1.0 / (1.0 + np.exp(bler_slope * (sinr - bler_mid)))
    bler = np.clip(bler_clean + rng.normal(0.0, 0.03, size=t), 0.0, 1.0)

    # HARQ retransmission count: higher when BLER is high, capped at 3.
    lam = 0.15 + 3.2 * bler
    harq = np.clip(rng.poisson(lam), 0, 3).astype(np.float64)

    return sinr, bler, harq


def build_dataset(n_sessions=360, window_len=WINDOW_LEN, stride=4, seed=SEED):
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for _ in range(n_sessions):
        sinr, bler, harq = _simulate_session(rng)
        t = len(sinr)
        last_start = t - 2 * window_len
        for start in range(0, last_start, stride):
            w = np.stack([
                sinr[start:start + window_len],
                bler[start:start + window_len],
                harq[start:start + window_len],
            ], axis=0)
            nxt_sinr = sinr[start + window_len:start + 2 * window_len]
            nxt_bler = bler[start + window_len:start + 2 * window_len]
            degraded = (nxt_sinr.mean() < SINR_DEGRADED_DB) or (nxt_bler.mean() > BLER_DEGRADED)
            windows.append(w)
            labels.append(1.0 if degraded else 0.0)
    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y


def normalize(X, stats=None):
    """Per-channel (SINR, BLER, HARQ) mean/std normalization."""
    if stats is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        stats = (mean, std)
    mean, std = stats
    return (X - mean) / std, stats


def split(X, y, seed=SEED, train_frac=0.70, val_frac=0.15):
    rng = np.random.default_rng(seed + 1)
    n = len(y)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


if __name__ == "__main__":
    X, y = build_dataset()
    print("windows:", X.shape, "positive frac:", y.mean())
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    print("train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)
    np.savez_compressed(
        "data/windows.npz",
        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
    )
    print("wrote data/windows.npz")
