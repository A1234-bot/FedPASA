"""FedPASA: federated patch-statistics aggregation for multi-label recognition."""

from .core import (
    Adapter,
    ClientPacket,
    build_client_banks,
    fit_central_piaa,
    fit_fedpasa,
    fit_fedpasa_from_packets,
    fit_quantile_aligned_fedpasa,
    make_client_packet,
    mean_average_precision,
    split_global_bank_by_client,
)
from .supervised import RidgePacket, fit_ridge_from_packets, make_ridge_packet

__all__ = [
    "Adapter",
    "ClientPacket",
    "build_client_banks",
    "fit_central_piaa",
    "fit_fedpasa",
    "fit_fedpasa_from_packets",
    "fit_quantile_aligned_fedpasa",
    "make_client_packet",
    "mean_average_precision",
    "split_global_bank_by_client",
    "RidgePacket",
    "fit_ridge_from_packets",
    "make_ridge_packet",
]
