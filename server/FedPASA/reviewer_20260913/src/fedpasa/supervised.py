from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .core import Adapter


@dataclass(frozen=True)
class RidgePacket:
    """Additive label-aware statistics; contains no individual example."""

    count: int
    sum_x: torch.Tensor  # [D]
    sum_y: torch.Tensor  # [C]
    xtx: torch.Tensor  # [D, D]
    xty: torch.Tensor  # [D, C]


def make_ridge_packet(features: np.ndarray, targets: np.ndarray) -> RidgePacket:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    # Accumulate the sufficient statistics in float64.  With small ridge
    # penalties the normal equations are ill-conditioned, and summing
    # client-local float32 Gram matrices can otherwise create visible drift.
    x = x.astype(np.float64)
    return RidgePacket(
        count=len(x),
        sum_x=torch.from_numpy(x.sum(axis=0)),
        sum_y=torch.from_numpy(y.sum(axis=0)),
        xtx=torch.from_numpy(x.T @ x),
        xty=torch.from_numpy(x.T @ y),
    )


def fit_ridge_from_packets(
    packets: tuple[RidgePacket, ...],
    *,
    regularization: float,
) -> Adapter:
    if not packets:
        raise ValueError("at least one ridge packet is required")
    count = sum(packet.count for packet in packets)
    dimension = packets[0].sum_x.numel()
    classes = packets[0].sum_y.numel()
    sum_x = sum((packet.sum_x for packet in packets), torch.zeros(dimension, dtype=torch.float64))
    sum_y = sum((packet.sum_y for packet in packets), torch.zeros(classes, dtype=torch.float64))
    xtx = sum(
        (packet.xtx for packet in packets),
        torch.zeros((dimension, dimension), dtype=torch.float64),
    )
    xty = sum(
        (packet.xty for packet in packets),
        torch.zeros((dimension, classes), dtype=torch.float64),
    )
    mean_x = sum_x / count
    mean_y = sum_y / count
    centered_xtx = xtx - count * torch.outer(mean_x, mean_x)
    centered_xty = xty - count * torch.outer(mean_x, mean_y)
    scale = torch.trace(centered_xtx).clamp_min(1e-12) / dimension
    system = centered_xtx + regularization * scale * torch.eye(dimension, dtype=torch.float64)
    weight = torch.linalg.solve(system, centered_xty)
    bias = mean_y - mean_x @ weight
    packet_floats = dimension * dimension + dimension * classes + dimension + classes + 1
    communication_bytes = len(packets) * packet_floats * 8
    return Adapter(
        weight.float(),
        bias.float(),
        {
            "method": "label_aware_federated_ridge_statistics",
            "clients": len(packets),
            "regularization": regularization,
            "regularization_scale": float(scale),
            "individual_examples_transmitted": False,
            "statistics_dtype": "float64",
            "estimated_communication_MiB": communication_bytes / 1024**2,
        },
    )
