"""Strict block scoring of the rev8k campaign and stationary error diagnostics.

* :func:`strict_score` is the executed v1 scorer (``rev8k-strict-v1``), unchanged: predictions and targets are aligned
  by record id; missing, extra or duplicate ids, a shape mismatch and non-finite prediction values make the block
  invalid (a reason is given, nothing is skipped or clipped). Each record's error is the float64 relative L2
  ||p - t|| / max(||t||, 1e-12), joint over all target frames. Rollout blocks (5-d arrays) also get the mean error per
  horizon; the joint metric stays primary.
* :func:`stationary_diagnostics` (contract ``pdeobs.rev8k.stationary/1``) splits the error of a static recovery block
  [N, H, W] with its observation masks, following the handoff reference helper: raw relative L2, the observed and hidden
  contributions (each normalised by ||target||, so raw^2 = observed^2 + hidden^2), and the data-consistent error after
  pasting the observed target cells back into the prediction.
"""
from __future__ import annotations

import numpy as np

SCORER_VERSION = "rev8k-strict-v1"
STATIONARY_CONTRACT = "pdeobs.rev8k.stationary/1"
EPS = 1e-12


def strict_score(pred_ids, pred, target_ids, target, expected_ids) -> dict:
    """Contract checks + float64 per-record relative L2 aligned by id."""
    out = {"scorer_version": SCORER_VERSION, "expected_n": len(expected_ids)}
    if len(set(expected_ids)) != len(expected_ids):
        return dict(out, valid=False, reason="expected id list has duplicates")
    if len(pred_ids) != len(set(pred_ids)):
        return dict(out, valid=False, reason="duplicate ids in the prediction block")
    if set(pred_ids) != set(expected_ids):
        missing, extra = set(expected_ids) - set(pred_ids), set(pred_ids) - set(expected_ids)
        return dict(out, valid=False, reason="id mismatch", missing=sorted(missing)[:5], extra=sorted(extra)[:5],
                    n_missing=len(missing), n_extra=len(extra))
    if set(target_ids) != set(expected_ids) or len(target_ids) != len(expected_ids):
        return dict(out, valid=False, reason="target id mismatch")
    t_index = {i: k for k, i in enumerate(target_ids)}
    p_index = {i: k for k, i in enumerate(pred_ids)}
    p = pred[[p_index[i] for i in expected_ids]].astype(np.float64)
    t = target[[t_index[i] for i in expected_ids]]
    if p.shape != t.shape:
        return dict(out, valid=False, reason="shape mismatch %s vs %s" % (p.shape, t.shape))
    if not np.isfinite(p).all():
        return dict(out, valid=False, reason="non-finite prediction values", n_nonfinite_records=int((~np.isfinite(p.reshape(len(p), -1))).any(1).sum()))
    n = len(expected_ids)
    num = np.sqrt(((p - t) ** 2).reshape(n, -1).sum(1))
    den = np.maximum(np.sqrt((t ** 2).reshape(n, -1).sum(1)), EPS)
    e = num / den
    res = dict(out, valid=True, n=n, mean=float(e.mean()), median=float(np.median(e)), p90=float(np.quantile(e, 0.9)),
               max=float(e.max()), per_record=[[i, float(v)] for i, v in zip(expected_ids, e)])
    if p.ndim == 5:                                       # rollout: per-horizon errors (the joint metric stays primary)
        num_h = np.sqrt(((p - t) ** 2).reshape(n, p.shape[1], -1).sum(2))
        den_h = np.maximum(np.sqrt((t ** 2).reshape(n, t.shape[1], -1).sum(2)), EPS)
        res["mean_per_horizon"] = [float(x) for x in (num_h / den_h).mean(0)]
    return res


def relative_l2(pred, target) -> np.ndarray:
    """Per-record float64 relative L2 ||p - y|| / max(||y||, 1e-12) over a leading record axis; equal shapes and finite
    values required (a non-finite value is an error, never skipped)."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim < 2 or p.shape[0] == 0:
        raise ValueError("relative_l2 needs equal shapes with a leading record axis")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("non-finite prediction/target values; they are not skipped")
    n = p.shape[0]
    d = np.maximum(np.linalg.norm(y.reshape(n, -1), axis=1), EPS)
    return np.linalg.norm((p - y).reshape(n, -1), axis=1) / d


def stationary_diagnostics(pred, target, mask, *, return_prediction: bool = False) -> dict:
    """Error split of a static recovery block. pred, target: [N, H, W]; mask: boolean [N, H, W] (True = observed).

    Per record (float64 arrays of length N):
      raw              ||p - y|| / ||y||
      observed         ||M (p - y)|| / ||y||          (observed contribution)
      hidden           ||(1 - M)(p - y)|| / ||y||     (hidden-only contribution)
      data_consistent  ||q - y|| / ||y||, q = p with the observed target cells pasted back
    with ||y|| floored at 1e-12. These are the handoff reference quantities ``raw``, ``observed_contribution``,
    ``hidden_contribution`` and ``projected_error``. ``return_prediction`` adds q as ``data_consistent_prediction``."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    m = np.asarray(mask)
    if m.dtype != bool or p.shape != y.shape or m.shape != p.shape:
        raise ValueError("stationary diagnostics need pred, target and a boolean mask of one shape")
    if p.ndim != 3:
        raise ValueError("stationary diagnostics expect [N, H, W] arrays")
    raw = relative_l2(p, y)
    n = p.shape[0]
    d = np.maximum(np.linalg.norm(y.reshape(n, -1), axis=1), EPS)
    observed = np.linalg.norm((m * (p - y)).reshape(n, -1), axis=1) / d
    hidden = np.linalg.norm(((~m) * (p - y)).reshape(n, -1), axis=1) / d
    q = np.where(m, y, p)
    out = {"contract": STATIONARY_CONTRACT, "n": int(n), "raw": raw, "observed": observed, "hidden": hidden,
           "data_consistent": relative_l2(q, y)}
    if return_prediction:
        out["data_consistent_prediction"] = q
    return out


def summarize(values) -> dict:
    """mean / median / p90 / max of a per-record error array."""
    v = np.asarray(values, dtype=np.float64)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
            "max": float(v.max())}


__all__ = ["EPS", "SCORER_VERSION", "STATIONARY_CONTRACT", "relative_l2", "stationary_diagnostics", "strict_score",
           "summarize"]
