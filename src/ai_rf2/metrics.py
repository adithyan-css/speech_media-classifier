"""Window-level metrics in the SSOW vocabulary.

* Detection rate (DR)  = SPEECH windows scored SPEECH / all SPEECH windows (recall).
* False alarm rate (FAR) = MEDIA windows scored SPEECH / all MEDIA windows.

Event-level DR / FAR (SSOW §2.5, PRD 7.2) are computed by ``aacara.evaluate`` on
continuous streams - see ``ai_rf2.stream_eval``.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def threshold_for_far(logits: np.ndarray, labels: np.ndarray, far: float = 0.05) -> float:
    """Smallest logit threshold whose false alarm rate is <= ``far`` (score >= thr => SPEECH)."""
    neg = np.sort(np.asarray(logits)[np.asarray(labels) == 0])
    if neg.size == 0:
        return float("-inf")
    k = int(np.floor(far * neg.size))            # at most k negatives may reach the threshold
    thr = neg[neg.size - k - 1] if k < neg.size else neg[0]
    return float(np.nextafter(thr, np.inf))


def threshold_max_margin(logits: np.ndarray, labels: np.ndarray, dr_min: float = 0.95,
                         far_max: float = 0.05) -> float:
    """The threshold that maximises the *smaller* SSOW margin, min(DR - 0.95, 0.05 - FAR).

    Spending the whole 5 % false-alarm budget to squeeze out the last 0.3 % of detection
    leaves no margin on FAR, and a held-out set then lands on either side of 5 %.  This
    rule centres the operating point between the two acceptance limits instead.
    """
    logits, labels = np.asarray(logits, np.float64), np.asarray(labels).astype(int)
    cands = np.unique(logits)
    order = np.argsort(logits)
    s, y = logits[order], labels[order]
    n_pos, n_neg = max(1, y.sum()), max(1, (y == 0).sum())
    # for threshold = s[i]: predicted positive are indices >= i
    pos_below = np.concatenate([[0], np.cumsum(y)])          # positives strictly below i
    neg_below = np.concatenate([[0], np.cumsum(1 - y)])
    first = np.searchsorted(s, cands, side="left")
    dr = 1 - pos_below[first] / n_pos
    far = 1 - neg_below[first] / n_neg
    margin = np.minimum(dr - dr_min, far_max - far)
    return float(cands[int(np.argmax(margin))])


def at_threshold(logits: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    logits, labels = np.asarray(logits), np.asarray(labels).astype(int)
    pred = logits >= thr
    tp = int(np.sum(pred & (labels == 1)))
    fn = int(np.sum(~pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    tn = int(np.sum(~pred & (labels == 0)))
    dr = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (fp + tn) if fp + tn else float("nan")
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    bal = np.nanmean([dr, 1 - far]) if not (np.isnan(dr) and np.isnan(far)) else float("nan")
    return {"threshold_logit": float(thr), "threshold_prob": float(sigmoid(thr)),
            "dr": dr, "far": far, "accuracy": acc, "balanced_accuracy": float(bal),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def window_metrics(logits: np.ndarray, labels: np.ndarray, thr: float | None = None) -> dict:
    """AUC, DR at FAR = 5 %, and the confusion at 0.5 (logit 0) and at ``thr`` if given."""
    logits, labels = np.asarray(logits, np.float64), np.asarray(labels).astype(int)
    both = len(np.unique(labels)) == 2
    out = {"n": int(labels.size), "n_speech": int(labels.sum()),
           "n_media": int((labels == 0).sum()),
           "auc": float(roc_auc_score(labels, logits)) if both else float("nan")}
    if both:
        t5 = threshold_for_far(logits, labels, 0.05)
        out["dr_at_far5"] = at_threshold(logits, labels, t5)["dr"]
        t1 = threshold_for_far(logits, labels, 0.01)
        out["dr_at_far1"] = at_threshold(logits, labels, t1)["dr"]
    else:
        out["dr_at_far5"] = out["dr_at_far1"] = float("nan")
    at05 = at_threshold(logits, labels, 0.0)
    out["balanced_accuracy@0.5"] = at05["balanced_accuracy"]
    out["accuracy@0.5"] = at05["accuracy"]
    out["dr@0.5"], out["far@0.5"] = at05["dr"], at05["far"]
    if thr is not None:
        op = at_threshold(logits, labels, thr)
        out["operating_point"] = op
    return out


def roc(logits: np.ndarray, labels: np.ndarray):
    fpr, tpr, thr = roc_curve(np.asarray(labels).astype(int), np.asarray(logits))
    return fpr, tpr, thr
