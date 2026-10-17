"""Train the link-degradation CNN, pick an operating threshold, measure precision.

Genuine optimization attempts toward the 0.91 precision claim: controlled by
ATTEMPT (1..3) which widens the model / training budget slightly each time.
The threshold is chosen on the validation split only (never on test), to hit
as close to 0.91 precision as the validation precision-recall curve allows;
test precision is then measured once at that fixed threshold.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")
from datagen import build_dataset, normalize, split  # noqa: E402
from model import LinkDegradationCNN  # noqa: E402

TARGET_PRECISION = 0.91


def precision_recall_at_thresholds(probs, y):
    order = np.argsort(-probs)
    probs_s, y_s = probs[order], y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    return probs_s, precision, recall


def pick_threshold_for_precision(probs, y, target, min_predicted_positive=20):
    probs_s, precision, recall = precision_recall_at_thresholds(probs, y)
    n_predicted = np.arange(1, len(probs_s) + 1)
    reliable = n_predicted >= min_predicted_positive
    ok = np.where((precision >= target) & reliable)[0]
    if len(ok) == 0:
        cand = np.where(reliable)[0]
        if len(cand) == 0:
            cand = np.arange(len(probs_s))
        best_i = cand[int(np.argmax(precision[cand]))]
        return float(probs_s[best_i]), float(precision[best_i]), float(recall[best_i])
    i = ok[np.argmax(recall[ok])]
    return float(probs_s[i]), float(precision[i]), float(recall[i])


def train_once(Xtr, ytr, Xva, yva, epochs, c1, c2, c3, seed):
    torch.manual_seed(seed)
    model = LinkDegradationCNN(c1=c1, c2=c2, c3=c3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)

    n = len(ytr_t)
    batch = 128
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logit = model(Xtr_t[idx])
            loss = loss_fn(logit, ytr_t[idx])
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                va_probs = torch.sigmoid(model(Xva_t)).numpy()
            va_prec, va_rec = quick_pr(va_probs, yva)
            print(f"epoch {epoch:3d} loss {tot_loss / n:.4f} val_prec@0.5 {va_prec:.3f} val_rec@0.5 {va_rec:.3f}")
    return model


def quick_pr(probs, y, thresh=0.5):
    pred = (probs >= thresh).astype(np.float32)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec


def main():
    X, y = build_dataset()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    Xtr_n, stats = normalize(Xtr)
    Xva_n, _ = normalize(Xva, stats)
    Xte_n, _ = normalize(Xte, stats)

    attempts = []
    configs = [
        dict(epochs=18, c1=16, c2=64, c3=64, seed=0),
        dict(epochs=30, c1=16, c2=64, c3=64, seed=1),
        dict(epochs=30, c1=24, c2=80, c3=80, seed=2),
    ]
    best = None
    for i, cfg in enumerate(configs, start=1):
        print(f"=== attempt {i}/3: {cfg} ===")
        model = train_once(Xtr_n, ytr, Xva_n, yva, **cfg)
        model.eval()
        with torch.no_grad():
            va_probs = torch.sigmoid(model(torch.from_numpy(Xva_n))).numpy()
        thresh, va_prec, va_rec = pick_threshold_for_precision(va_probs, yva, TARGET_PRECISION)
        attempts.append(dict(attempt=i, config=cfg, val_precision=va_prec, val_recall=va_rec, threshold=thresh))
        print(f"  val precision at chosen threshold: {va_prec:.4f} (target {TARGET_PRECISION}), threshold={thresh:.4f}")
        if best is None or va_prec > best[1]:
            best = (model, va_prec, thresh, stats, cfg)
        if va_prec >= TARGET_PRECISION:
            break

    model, _, thresh, stats, cfg = best
    model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(model(torch.from_numpy(Xte_n))).numpy()
    te_pred = (te_probs >= thresh).astype(np.float32)
    tp = int(((te_pred == 1) & (yte == 1)).sum())
    fp = int(((te_pred == 1) & (yte == 0)).sum())
    fn = int(((te_pred == 0) & (yte == 1)).sum())
    test_precision = tp / max(tp + fp, 1)
    test_recall = tp / max(tp + fn, 1)

    torch.save(model.state_dict(), "data/model.pt")
    np.savez("data/norm_stats.npz", mean=stats[0], std=stats[1])
    report = dict(
        attempts=attempts,
        chosen_config=cfg,
        threshold=float(thresh),
        test_precision=test_precision,
        test_recall=test_recall,
        test_support=dict(tp=tp, fp=fp, fn=fn, n=int(len(yte)), positives=int(yte.sum())),
        target_precision=TARGET_PRECISION,
        meets_claim=bool(test_precision >= TARGET_PRECISION),
        weight_param_count=model.weight_param_count(),
    )
    with open("docs/train_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
