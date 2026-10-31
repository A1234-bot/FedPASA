from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fedpasa.core as core  # noqa: E402
from fedpasa.supervised import fit_ridge_from_packets, make_ridge_packet  # noqa: E402


def synthetic_banks() -> tuple[torch.Tensor, tuple[dict[int, core.PatchBank], ...]]:
    torch.manual_seed(4)
    text = F.normalize(torch.randn(3, 8), dim=-1)
    clients = []
    for _ in range(2):
        bank = {}
        for class_id in range(3):
            features = F.normalize(text[class_id] + 0.12 * torch.randn(10, 8), dim=-1)
            probabilities = torch.softmax(features @ text.T * 15.0, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1)
            bank[class_id] = core.PatchBank(features, probabilities, entropy)
        clients.append(bank)
    return text, tuple(clients)


def test_quantile_protocol_recovers_central_adapter() -> None:
    text, clients = synthetic_banks()
    merged = {}
    for class_id in range(3):
        for bank in clients:
            value = bank[class_id]
            merged[class_id] = core._merge_topk(
                merged.get(class_id),
                value.features,
                value.probabilities,
                value.entropy,
                8,
            )
    central = core.fit_central_piaa(merged, text)
    federated = core.fit_quantile_aligned_fedpasa(clients, text, k=8)
    torch.testing.assert_close(federated.weight, central.weight, atol=1e-5, rtol=0)
    torch.testing.assert_close(federated.bias, central.bias, atol=1e-5, rtol=0)
    assert federated.metadata["individual_patch_embeddings_transmitted"] is False
    assert federated.metadata["rounds"] == 4
    assert federated.metadata["purification_packet_floats_per_client"] == 9


def test_distributed_purification_threshold_matches_central_sample_std() -> None:
    refined = torch.tensor(
        [
            [0.80, 0.15, 0.05],
            [0.70, 0.20, 0.10],
            [0.10, 0.65, 0.25],
            [0.05, 0.60, 0.35],
            [0.10, 0.20, 0.70],
        ],
        dtype=torch.float64,
    )
    predictions = torch.tensor([0, 0, 1, 1, 2])
    packets = (
        core._purification_packet(refined[:2], predictions[:2], classes=3),
        core._purification_packet(refined[2:], predictions[2:], classes=3),
    )
    distributed = core._purification_threshold_from_packets(packets, classes=3)

    expected = torch.empty(3, dtype=torch.float64)
    for class_id in range(3):
        confidence = refined[predictions == class_id, class_id]
        deviation = confidence.std(unbiased=True) if confidence.numel() > 1 else 0.0
        expected[class_id] = confidence.mean() + deviation
    torch.testing.assert_close(distributed, expected, atol=1e-12, rtol=0)


def test_mean_average_precision_is_percentage() -> None:
    targets = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.uint8)
    scores = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    mean_ap, per_class = core.mean_average_precision(targets, scores)
    assert mean_ap == 100.0
    np.testing.assert_allclose(per_class, [100.0, 100.0])


def test_split_global_bank_preserves_selected_patches() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    probabilities = torch.softmax(torch.randn(4, 3), dim=-1)
    entropy = torch.tensor([0.4, 0.1, 0.3, 0.2])
    origins = torch.tensor([0, 1, 2, 3])
    global_bank = {1: core.PatchBank(features, probabilities, entropy, origins)}
    clients = core.split_global_bank_by_client(global_bank, np.asarray([0, 1, 0, 1]), 2)
    assert sum(len(bank[1].features) for bank in clients) == 4
    recovered = torch.cat([bank[1].image_indices for bank in clients]).sort().values
    torch.testing.assert_close(recovered, origins)


def test_label_ridge_packets_recover_central_solution() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(80, 12)).astype(np.float32)
    targets = (rng.random(size=(80, 5)) > 0.65).astype(np.uint8)
    central = fit_ridge_from_packets((make_ridge_packet(features, targets),), regularization=0.3)
    distributed = fit_ridge_from_packets(
        (
            make_ridge_packet(features[:31], targets[:31]),
            make_ridge_packet(features[31:57], targets[31:57]),
            make_ridge_packet(features[57:], targets[57:]),
        ),
        regularization=0.3,
    )
    torch.testing.assert_close(distributed.weight, central.weight, atol=2e-6, rtol=0)
    torch.testing.assert_close(distributed.bias, central.bias, atol=2e-6, rtol=0)
