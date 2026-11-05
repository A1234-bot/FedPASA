"""Exact Full calibration from client messages containing sample-level data.

Only the client encoder accepts local scores/labels. The server takes serialized
messages, restores validation order, and performs the historical float32 z-score
and exact AP search. This is not a sufficient-statistics or privacy protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import numpy as np

from .core import mean_average_precision


def encode_calibration(positions, qa_scores, ridge_scores, labels) -> bytes:
    """Encode one client's validation rows; positions are split-local ordinals."""
    positions = np.asarray(positions)
    qa, ridge, y = map(np.asarray, (qa_scores, ridge_scores, labels))
    if qa.ndim != 2 or qa.shape != ridge.shape or qa.shape != y.shape:
        raise ValueError("scores and labels must have identical N x C shapes")
    if positions.shape != (len(qa),) or positions.dtype.kind not in "iu":
        raise ValueError("positions must be an integer vector matching the rows")
    if np.any(positions < 0) or len(np.unique(positions)) != len(positions):
        raise ValueError("positions must be unique and nonnegative")
    if qa.dtype != np.float32 or ridge.dtype != np.float32:
        raise ValueError("historical Full requires float32 scores; no silent quantization")
    if not np.isfinite(qa).all() or not np.isfinite(ridge).all():
        raise ValueError("scores must be finite")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    buffer = BytesIO()
    np.savez(buffer, positions=positions.astype("<i8"), qa=qa, ridge=ridge,
             labels=y.astype(np.uint8))
    return buffer.getvalue()


@dataclass
class FullCalibration:
    qa: np.ndarray
    ridge: np.ndarray
    weights: np.ndarray
    grid: tuple[float, ...]
    validation_ap: np.ndarray
    messages: list[dict]

    @property
    def global_weight(self) -> float:
        return self.grid[int(self.validation_ap.mean(axis=1).argmax())]

    @property
    def uplink_bytes(self) -> int:
        return sum(message["wire_bytes"] for message in self.messages)

    def predict(self, qa_scores, ridge_scores, *, global_weight=False):
        q = (qa_scores - self.qa[0]) / self.qa[1]
        r = (ridge_scores - self.ridge[0]) / self.ridge[1]
        w = self.global_weight if global_weight else self.weights[None, :]
        return w * r + (1.0 - w) * q


def calibrate_from_messages(messages: Iterable[bytes], *, expected_rows: int,
                            classes: int, weights=None) -> FullCalibration:
    """Server API: no feature cache, labels, assignments or test data access."""
    if expected_rows <= 0 or classes <= 0:
        raise ValueError("calibration requires positive rows and classes")
    grid = tuple(float(w) for w in (np.linspace(0, 1, 21) if weights is None else weights))
    if not grid or not np.isfinite(grid).all() or min(grid) < 0 or max(grid) > 1:
        raise ValueError("weights must be finite and in [0, 1]")
    qa = np.empty((expected_rows, classes), np.float32)
    ridge = np.empty_like(qa)
    labels = np.empty(qa.shape, np.uint8)
    seen = np.zeros(expected_rows, bool)
    ledger = []
    for client, wire in enumerate(messages):
        with np.load(BytesIO(wire), allow_pickle=False) as packet:
            if set(packet.files) != {"positions", "qa", "ridge", "labels"}:
                raise ValueError("unexpected calibration fields")
            pos, q, r, y = (packet[k] for k in ("positions", "qa", "ridge", "labels"))
        # Apply the same validation to untrusted incoming packets.
        encode_calibration(pos, q, r, y)
        if q.shape[1] != classes or np.any(pos >= expected_rows) or seen[pos].any():
            raise ValueError("invalid, duplicate or overlapping validation rows")
        qa[pos], ridge[pos], labels[pos] = q, r, y
        seen[pos] = True
        fields = [{"name": k, "shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes}
                  for k, a in zip(("positions", "qa", "ridge", "labels"), (pos, q, r, y))]
        ledger.append({"client": client, "direction": "uplink", "stage": "exact_full_calibration",
                       "rows": len(pos), "fields": fields, "wire_bytes": len(wire),
                       "numeric_bytes": sum(f["bytes"] for f in fields)})
    if not seen.all():
        raise ValueError("missing validation rows")
    qc = np.stack((qa.mean(0), qa.std(0).clip(min=1e-6)))
    rc = np.stack((ridge.mean(0), ridge.std(0).clip(min=1e-6)))
    qz, rz = (qa - qc[0]) / qc[1], (ridge - rc[0]) / rc[1]
    ap = np.stack([mean_average_precision(labels, w * rz + (1.0 - w) * qz)[1] for w in grid])
    chosen = np.asarray(grid, np.float32)[ap.argmax(0)]
    return FullCalibration(qc, rc, chosen, grid, ap, ledger)


def simulate_calibration(qa_scores, ridge_scores, labels, owners, *, clients, weights=None):
    """Single-host transport harness; each encoder sees only its client's rows."""
    owners = np.asarray(owners)
    if owners.shape != (len(labels),) or owners.dtype.kind not in "iu" or clients <= 0:
        raise ValueError("owners must assign every validation row to a client")
    if np.any(owners < 0) or np.any(owners >= clients):
        raise ValueError("owner outside client range")
    def messages():
        for client in range(clients):
            pos = np.flatnonzero(owners == client)
            yield encode_calibration(pos, qa_scores[pos], ridge_scores[pos], labels[pos])
    return calibrate_from_messages(messages(), expected_rows=len(labels),
                                   classes=labels.shape[1], weights=weights)
