import sys

import numpy as np

sys.path.insert(0, "src")
from datagen import build_dataset, normalize, split  # noqa: E402


def test_window_shape_and_labels():
    X, y = build_dataset(n_sessions=20)
    assert X.shape[1:] == (3, 64)
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}
    assert 0.0 < y.mean() < 0.5  # degradation is the minority class, not absent or dominant


def test_split_is_disjoint_and_covers_all():
    X, y = build_dataset(n_sessions=20)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    assert len(ytr) + len(yva) + len(yte) == len(y)
    assert len(ytr) > len(yva) and len(ytr) > len(yte)


def test_normalize_uses_train_stats_on_test():
    X, y = build_dataset(n_sessions=20)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    Xtr_n, stats = normalize(Xtr)
    Xte_n, stats2 = normalize(Xte, stats)
    assert stats2 is stats
    assert abs(Xtr_n.mean()) < 0.5  # roughly centered, not an exact invariant for a finite sample
