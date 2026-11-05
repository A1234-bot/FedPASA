from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from . import gpu_ops
from . import cache_io


@dataclass(frozen=True)
class Adapter:
    """Closed-form GDA adapter in the same orientation as the PIAA release."""

    weight: torch.Tensor  # [D, C]
    bias: torch.Tensor  # [C]
    metadata: dict[str, object]


@dataclass(frozen=True)
class PatchBank:
    features: torch.Tensor  # [N, D], CPU float32
    probabilities: torch.Tensor  # [N, C], CPU float32
    entropy: torch.Tensor  # [N], CPU float32
    image_indices: torch.Tensor | None = None  # [N], source image for protocol sweeps
    patch_ids: torch.Tensor | None = None  # [N], shared image-index * patches-per-image + offset


@dataclass(frozen=True)
class ClientPacket:
    """The only information a FedPASA client transmits.

    Class-wise vectors are O(CD); pooled scatter is O(D^2) once per client.
    No image, label, patch identity, or individual patch embedding is transmitted.
    """

    count: torch.Tensor  # [C]
    mass: torch.Tensor  # [C]
    mean: torch.Tensor  # [C, D]
    entropy: torch.Tensor  # [C], normalized predictive entropy
    scatter: torch.Tensor  # [D, D], pooled within-class scatter
    scatter_dof: int


@dataclass(frozen=True)
class MomentPacket:
    """Mergeable class moments used by the quantile-aligned protocol."""

    count: torch.Tensor  # [C]
    sum_x: torch.Tensor  # [C, D]
    mass: torch.Tensor  # [C]
    weighted_sum_x: torch.Tensor  # [C, D]
    second_moment: torch.Tensor  # [D, D], shared across classes


@dataclass(frozen=True)
class PurificationPacket:
    """Client-local confidence moments for the global purification threshold.

    The squared-confidence sum is irreducible: it cannot be reconstructed from
    ``count`` and ``sum_q``. Keeping it in an explicit packet also prevents a
    federated implementation from reading per-patch confidences at the server.
    """

    count: torch.Tensor  # [C]
    sum_q: torch.Tensor  # [C]
    sum_q2: torch.Tensor  # [C]


def mean_average_precision(targets: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray]:
    targets = np.asarray(targets)
    scores = np.asarray(scores)
    if (gpu_ops.enabled() and targets.shape == scores.shape and len(scores)
            and np.isin(targets, (0, 1)).all() and np.isfinite(scores).all()):
        return gpu_ops.average_precision(targets, scores)
    per_class = np.asarray(
        [average_precision_score(targets[:, c], scores[:, c]) for c in range(scores.shape[1])],
        dtype=np.float64,
    )
    return float(100.0 * per_class.mean()), 100.0 * per_class


def _merge_topk(
    current: PatchBank | None,
    features: torch.Tensor,
    probabilities: torch.Tensor,
    entropy: torch.Tensor,
    k: int,
    image_indices: torch.Tensor | None = None,
) -> PatchBank:
    features = features.detach().float().cpu()
    probabilities = probabilities.detach().float().cpu()
    entropy = entropy.detach().float().cpu()
    origins = None if image_indices is None else image_indices.detach().long().cpu()
    if current is not None:
        features = torch.cat((current.features, features), dim=0)
        probabilities = torch.cat((current.probabilities, probabilities), dim=0)
        entropy = torch.cat((current.entropy, entropy), dim=0)
        if current.image_indices is not None and origins is not None:
            origins = torch.cat((current.image_indices, origins), dim=0)
        else:
            origins = None
    keep = min(k, entropy.numel())
    indices = torch.topk(entropy, keep, largest=False, sorted=False).indices
    kept_origins = None if origins is None else origins[indices]
    return PatchBank(features[indices], probabilities[indices], entropy[indices], kept_origins)


@torch.inference_mode()
def build_client_banks(
    patch_cache: Path,
    assignments: np.ndarray,
    text_features: torch.Tensor,
    *,
    clients: int,
    k: int = 512,
    batch_images: int = 48,
    device: str = "cuda",
) -> tuple[dict[int, PatchBank], ...]:
    """Build client-local entropy top-k banks in one sequential cache pass."""

    patches = np.load(patch_cache, mmap_mode="r")
    assignments = np.asarray(assignments, dtype=np.int64)
    if patches.ndim != 3 or len(patches) != len(assignments):
        raise ValueError("patch cache and assignments must align as [images, patches, dim]")
    if assignments.min(initial=0) < -1 or assignments.max(initial=0) >= clients:
        raise ValueError("client assignments are out of range")

    text = F.normalize(text_features.to(device=device, dtype=torch.float32), dim=-1)
    banks: list[dict[int, PatchBank]] = [dict() for _ in range(clients)]
    for client_id in range(clients):
        image_indices = np.flatnonzero(assignments == client_id)
        if not len(image_indices):
            raise ValueError(f"client {client_id} owns zero images")
        for start, indices, (host_features,) in cache_io.batches((patches,), image_indices, batch_images):
            features = torch.from_numpy(host_features).to(device)
            features = F.normalize(features, dim=-1)
            flat = features.reshape(-1, features.shape[-1])
            logits = flat @ text.T
            probabilities = F.softmax(logits * 60.0, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum(dim=-1)
            predictions = logits.argmax(dim=-1)
            if gpu_ops.enabled() and flat.is_cuda:
                origins = torch.as_tensor(indices).repeat_interleave(features.shape[1])
                gpu_ops.merge_batch(banks[client_id], flat, probabilities, entropy,
                                    predictions, origins, k, PatchBank)
                del features, flat, logits, probabilities, entropy, predictions
                continue
            # Banks and top-k merges already live on CPU. Transfer each batch
            # once, then select classes there instead of repeatedly using CUDA
            # boolean indexing (the native crash site on the experiment host).
            flat = flat.cpu()
            probabilities = probabilities.cpu()
            entropy = entropy.cpu()
            predictions = predictions.cpu()
            origins = torch.as_tensor(indices).repeat_interleave(features.shape[1])
            for class_id in predictions.unique().tolist():
                mask = predictions == class_id
                banks[client_id][class_id] = _merge_topk(
                    banks[client_id].get(class_id),
                    flat[mask],
                    probabilities[mask],
                    entropy[mask],
                    k,
                    origins[mask],
                )
            del features, flat, logits, probabilities, entropy, predictions
        print(
            f"BANK client={client_id + 1}/{clients} images={len(image_indices)} "
            f"classes={len(banks[client_id])}",
            flush=True,
        )
    return tuple(banks)


def split_global_bank_by_client(
    global_bank: dict[int, PatchBank], assignments: np.ndarray, clients: int
) -> tuple[dict[int, PatchBank], ...]:
    """Partition a global top-k bank by source image without rescanning patch caches."""

    assignments = np.asarray(assignments, dtype=np.int64)
    output: list[dict[int, PatchBank]] = [dict() for _ in range(clients)]
    for class_id, value in global_bank.items():
        if value.image_indices is None:
            raise ValueError("global bank has no source-image indices")
        owners = torch.as_tensor(assignments[value.image_indices.numpy()], dtype=torch.long)
        for client_id in owners.unique().tolist():
            if client_id < 0:
                continue
            mask = owners == client_id
            output[client_id][class_id] = PatchBank(
                value.features[mask],
                value.probabilities[mask],
                value.entropy[mask],
                value.image_indices[mask],
                None if value.patch_ids is None else value.patch_ids[mask],
            )
    return tuple(output)


def _stack_banks(banks: Iterable[dict[int, PatchBank]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records: list[tuple[int, PatchBank]] = []
    for bank in banks:
        records.extend(sorted(bank.items()))
    if not records:
        raise ValueError("cannot stack empty patch banks")
    features = torch.cat([value.features for _, value in records], dim=0).double()
    probabilities = torch.cat([value.probabilities for _, value in records], dim=0).double()
    labels = torch.cat(
        [torch.full((len(value.features),), class_id, dtype=torch.long) for class_id, value in records]
    )
    return features, probabilities, labels


def _gda_from_samples(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the released PIAA confidence-weighted means and trace shrinkage."""

    features = features.double()
    probabilities = probabilities.double()
    text = F.normalize(text_features.double().cpu(), dim=-1)
    classes, dimension = text.shape
    means = text.clone()
    for class_id in range(classes):
        mask = labels == class_id
        if mask.any():
            weights = probabilities[mask, class_id].clamp_min(1e-12)
            means[class_id] = (weights[:, None] * features[mask]).sum(0) / weights.sum()

    centered = torch.cat(
        [features[labels == class_id] - means[class_id] for class_id in range(classes) if (labels == class_id).any()],
        dim=0,
    )
    dof = max(len(centered) - 1, 1)
    scatter = gpu_ops.matmul(centered.T, centered)
    trace_covariance = torch.trace(scatter) / dof
    regularized = scatter + trace_covariance * torch.eye(dimension, dtype=torch.float64)
    precision = dimension * gpu_ops.pinv(regularized)
    weight = precision @ means.T
    bias = -0.5 * torch.einsum("cd,de,ce->c", means, precision, means)
    bias = bias - np.log(classes)
    return weight, bias, means


def _moment_packet(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
) -> MomentPacket:
    features = features.double().cpu()
    probabilities = probabilities.double().cpu()
    labels = labels.long().cpu()
    dimension = features.shape[1]
    count = torch.zeros(classes, dtype=torch.float64)
    sum_x = torch.zeros((classes, dimension), dtype=torch.float64)
    mass = torch.zeros(classes, dtype=torch.float64)
    weighted_sum_x = torch.zeros((classes, dimension), dtype=torch.float64)
    for class_id in range(classes):
        mask = labels == class_id
        if not mask.any():
            continue
        class_features = features[mask]
        weights = probabilities[mask, class_id].clamp_min(1e-12)
        count[class_id] = len(class_features)
        sum_x[class_id] = class_features.sum(0)
        mass[class_id] = weights.sum()
        weighted_sum_x[class_id] = (weights[:, None] * class_features).sum(0)
    return MomentPacket(
        count=count,
        sum_x=sum_x,
        mass=mass,
        weighted_sum_x=weighted_sum_x,
        second_moment=gpu_ops.matmul(features.T, features),
    )


def _gda_from_moment_packets(
    packets: Iterable[MomentPacket], text_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packets = tuple(packets)
    if not packets:
        raise ValueError("at least one moment packet is required")
    text = F.normalize(text_features.double().cpu(), dim=-1)
    classes, dimension = text.shape
    count = sum((packet.count for packet in packets), torch.zeros(classes, dtype=torch.float64))
    sum_x = sum(
        (packet.sum_x for packet in packets),
        torch.zeros((classes, dimension), dtype=torch.float64),
    )
    mass = sum((packet.mass for packet in packets), torch.zeros(classes, dtype=torch.float64))
    weighted_sum_x = sum(
        (packet.weighted_sum_x for packet in packets),
        torch.zeros((classes, dimension), dtype=torch.float64),
    )
    second = sum(
        (packet.second_moment for packet in packets),
        torch.zeros((dimension, dimension), dtype=torch.float64),
    )
    means = text.clone()
    present = mass > 0
    means[present] = weighted_sum_x[present] / mass[present, None]
    scatter = second.clone()
    for class_id in range(classes):
        if count[class_id] <= 0:
            continue
        mean = means[class_id]
        total = sum_x[class_id]
        scatter -= torch.outer(mean, total)
        scatter -= torch.outer(total, mean)
        scatter += count[class_id] * torch.outer(mean, mean)
    dof = max(int(count.sum().item()) - 1, 1)
    trace_covariance = torch.trace(scatter) / dof
    regularized = scatter + trace_covariance * torch.eye(dimension, dtype=torch.float64)
    precision = dimension * gpu_ops.pinv(regularized)
    weight = precision @ means.T
    bias = -0.5 * torch.einsum("cd,de,ce->c", means, precision, means) - np.log(classes)
    return weight, bias, means


def _purification_packet(
    refined: torch.Tensor, predictions: torch.Tensor, classes: int
) -> PurificationPacket:
    """Build the three mergeable moments needed for per-class mean + sample std."""

    refined = refined.double().cpu()
    predictions = predictions.long().cpu()
    count = torch.zeros(classes, dtype=torch.float64)
    sum_q = torch.zeros(classes, dtype=torch.float64)
    sum_q2 = torch.zeros(classes, dtype=torch.float64)
    for class_id in predictions.unique().tolist():
        confidence = refined[predictions == class_id, class_id]
        count[class_id] = confidence.numel()
        sum_q[class_id] = confidence.sum()
        sum_q2[class_id] = confidence.square().sum()
    return PurificationPacket(count=count, sum_q=sum_q, sum_q2=sum_q2)


def _purification_threshold_from_packets(
    packets: Iterable[PurificationPacket], classes: int
) -> torch.Tensor:
    """Merge client packets into the global mean + unbiased standard deviation."""

    packets = tuple(packets)
    if not packets:
        raise ValueError("at least one purification packet is required")
    count = sum((packet.count for packet in packets), torch.zeros(classes, dtype=torch.float64))
    sum_q = sum((packet.sum_q for packet in packets), torch.zeros(classes, dtype=torch.float64))
    sum_q2 = sum((packet.sum_q2 for packet in packets), torch.zeros(classes, dtype=torch.float64))
    safe_count = count.clamp_min(1.0)
    mean = sum_q / safe_count
    variance = (sum_q2 - sum_q.square() / safe_count) / (count - 1.0).clamp_min(1.0)
    return mean + variance.clamp_min(0.0).sqrt()


def _global_quantile_selection(
    banks: tuple[dict[int, PatchBank], ...], k: int, *, shared_tie_key: bool = False
) -> tuple[dict[int, torch.Tensor], ...]:
    """Align per-class entropy top-k, with an optional common boundary-tie order.

    Exact IDs at a tied cutoff require every client's earlier local truncation
    to use the same (entropy, patch_id) order. A key added only at the server
    cannot recover candidates discarded by a different local tie rule.
    """

    selections: list[dict[int, torch.Tensor]] = [dict() for _ in banks]
    classes = sorted({class_id for bank in banks for class_id in bank})
    for class_id in classes:
        entropy_parts: list[torch.Tensor] = []
        owners: list[torch.Tensor] = []
        local_indices: list[torch.Tensor] = []
        key_parts: list[torch.Tensor] = []
        for client_id, bank in enumerate(banks):
            if class_id not in bank:
                continue
            entropy = bank[class_id].entropy
            entropy_parts.append(entropy)
            owners.append(torch.full((len(entropy),), client_id, dtype=torch.long))
            local_indices.append(torch.arange(len(entropy), dtype=torch.long))
            if shared_tie_key:
                if bank[class_id].patch_ids is None:
                    raise ValueError("shared tie ordering requires globally unique patch_ids")
                key_parts.append(bank[class_id].patch_ids)
        entropy = torch.cat(entropy_parts)
        owner = torch.cat(owners)
        local_index = torch.cat(local_indices)
        if shared_tie_key:
            keys = torch.cat(key_parts)
            if len(torch.unique(keys)) != len(keys):
                raise ValueError("patch_ids must be unique across clients within each class")
            by_key = torch.argsort(keys, stable=True)
            keep = by_key[torch.argsort(entropy[by_key], stable=True)[:k]]
        else:
            keep = torch.topk(entropy, min(k, len(entropy)), largest=False, sorted=False).indices
        for client_id in owner[keep].unique().tolist():
            client_mask = owner[keep] == client_id
            selections[client_id][class_id] = local_index[keep][client_mask]
    return tuple(selections)


def fit_quantile_aligned_fedpasa(
    banks: tuple[dict[int, PatchBank], ...],
    text_features: torch.Tensor,
    *,
    k: int = 512,
    purification_temperature: float = 100.0,
    shared_tie_key: bool = False,
) -> Adapter:
    """Recover global PIAA through four client uploads of mergeable statistics.

    Upload 1 aligns local entropy order statistics with a global per-class top-k.
    Upload 2 merges moments for the preliminary GDA. After that GDA is returned,
    upload 3 merges ``{count, sum(q), sum(q^2)}`` for the global purification
    threshold. Upload 4 merges moments of the purified patches for the final
    adapter. The third upload cannot be folded into the second because ``q``
    depends on the preliminary global GDA.

    ``shared_tie_key`` additionally uploads int64 patch IDs. Client-local banks
    must already have been truncated with the same lexicographic tie key.
    """

    if not banks:
        raise ValueError("at least one client bank is required")
    classes, dimension = text_features.shape
    selections = _global_quantile_selection(banks, k, shared_tie_key=shared_tie_key)
    local_selected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    preliminary_packets: list[MomentPacket] = []
    for bank, selection in zip(banks, selections):
        feature_parts: list[torch.Tensor] = []
        probability_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        for class_id, indices in sorted(selection.items()):
            value = bank[class_id]
            feature_parts.append(value.features[indices].double())
            probability_parts.append(value.probabilities[indices].double())
            label_parts.append(torch.full((len(indices),), class_id, dtype=torch.long))
        if not feature_parts:
            empty_features = torch.empty((0, dimension), dtype=torch.float64)
            empty_probabilities = torch.empty((0, classes), dtype=torch.float64)
            empty_labels = torch.empty(0, dtype=torch.long)
            local_selected.append((empty_features, empty_probabilities, empty_labels))
            continue
        features = torch.cat(feature_parts)
        probabilities = torch.cat(probability_parts)
        labels = torch.cat(label_parts)
        local_selected.append((features, probabilities, labels))
        preliminary_packets.append(_moment_packet(features, probabilities, labels, classes))

    preliminary_weight, preliminary_bias, _ = _gda_from_moment_packets(
        preliminary_packets, text_features
    )
    local_refined: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    purification_packets: list[PurificationPacket] = []
    for features, _, _ in local_selected:
        if not len(features):
            empty_refined = torch.empty((0, classes), dtype=torch.float64)
            empty_predictions = torch.empty(0, dtype=torch.long)
            local_refined.append(
                (features, empty_refined, empty_predictions)
            )
            purification_packets.append(
                _purification_packet(empty_refined, empty_predictions, classes)
            )
            continue
        logits = gpu_ops.matmul(features, preliminary_weight) + preliminary_bias
        refined = F.softmax(logits * purification_temperature, dim=-1)
        predictions = logits.argmax(dim=-1)
        local_refined.append((features, refined, predictions))
        purification_packets.append(_purification_packet(refined, predictions, classes))

    threshold = _purification_threshold_from_packets(purification_packets, classes)
    final_packets: list[MomentPacket] = []
    selected_count = 0
    for features, refined, predictions in local_refined:
        keep_parts: list[torch.Tensor] = []
        for class_id in predictions.unique().tolist():
            indices = torch.nonzero(predictions == class_id, as_tuple=False).flatten()
            keep_parts.append(indices[refined[indices, class_id] >= threshold[class_id]])
        if not keep_parts:
            continue
        keep = torch.cat(keep_parts)
        selected_count += len(keep)
        final_packets.append(
            _moment_packet(features[keep], refined[keep], predictions[keep], classes)
        )
    weight, bias, _ = _gda_from_moment_packets(final_packets, text_features)

    float_bytes = 4
    # Round 1 transmits local order statistics. C*K per client is a conservative
    # upper bound; missing local classes only reduce this number.
    entropy_scalars = len(banks) * classes * k
    moment_packet_floats = dimension * dimension + 2 * classes * dimension + 2 * classes
    purification_packet_floats = 3 * classes
    uplink_floats = (
        entropy_scalars
        + len(banks) * moment_packet_floats
        + len(banks) * purification_packet_floats
        + len(banks) * moment_packet_floats
    )
    tie_key_bytes = entropy_scalars * 8 if shared_tie_key else 0
    communication_bytes = float_bytes * uplink_floats + tie_key_bytes
    return Adapter(
        weight.float(),
        bias.float(),
        {
            "method": "quantile_aligned_fedpasa",
            "clients": len(banks),
            "rounds": 4,
            "client_uploads": 4,
            "global_topk": k,
            "shared_tie_key": shared_tie_key,
            "tie_key_uplink_bytes": tie_key_bytes,
            "selected_patches": selected_count,
            "individual_patch_embeddings_transmitted": False,
            "communication_scope": "client-to-server uploads only, dense FP32 upper bound",
            "purification_packet_floats_per_client": purification_packet_floats,
            "estimated_uplink_MiB": communication_bytes / 1024**2,
            "estimated_communication_MiB": communication_bytes / 1024**2,
        },
    )


def _purify_bank(
    bank: dict[int, PatchBank],
    text_features: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, probabilities, labels = _stack_banks((bank,))
    weight, bias, _ = _gda_from_samples(features, probabilities, labels, text_features)
    logits = gpu_ops.matmul(features, weight) + bias
    refined = F.softmax(logits * temperature, dim=-1)
    predictions = logits.argmax(dim=-1)
    keep_parts: list[torch.Tensor] = []
    for class_id in predictions.unique().tolist():
        indices = torch.nonzero(predictions == class_id, as_tuple=False).flatten()
        confidence = refined[indices, class_id]
        # torch.std's released default is unbiased=True.
        deviation = confidence.std(unbiased=True) if confidence.numel() > 1 else confidence.new_zeros(())
        threshold = confidence.mean() + deviation
        keep_parts.append(indices[confidence >= threshold])
    keep = torch.cat(keep_parts) if keep_parts else torch.arange(len(features))
    kept_predictions = predictions[keep]
    return features[keep], refined[keep], kept_predictions, keep


def fit_central_piaa(
    bank: dict[int, PatchBank],
    text_features: torch.Tensor,
    *,
    purification_temperature: float = 100.0,
) -> Adapter:
    features, probabilities, labels, _ = _purify_bank(
        bank, text_features, temperature=purification_temperature
    )
    weight, bias, _ = _gda_from_samples(features, probabilities, labels, text_features)
    return Adapter(
        weight.float(),
        bias.float(),
        {
            "method": "central_piaa_reproduction",
            "selected_patches": int(len(features)),
            "raw_patches": int(sum(len(value.features) for value in bank.values())),
        },
    )


def make_client_packet(
    bank: dict[int, PatchBank],
    text_features: torch.Tensor,
    *,
    purification_temperature: float = 100.0,
) -> ClientPacket:
    features, probabilities, labels, keep = _purify_bank(
        bank, text_features, temperature=purification_temperature
    )
    classes = text_features.shape[0]
    dimension = text_features.shape[1]
    count = torch.zeros(classes, dtype=torch.float64)
    mass = torch.zeros(classes, dtype=torch.float64)
    means = F.normalize(text_features.double().cpu(), dim=-1)
    normalized_entropy = torch.ones(classes, dtype=torch.float64)
    scatter = torch.zeros((dimension, dimension), dtype=torch.float64)

    # Recover the entropy ordering of the stacked raw bank so selected entropy is auditable.
    raw_entropy = torch.cat([value.entropy for _, value in sorted(bank.items())], dim=0).double()
    selected_entropy = raw_entropy[keep]
    log_classes = float(np.log(classes))
    for class_id in range(classes):
        mask = labels == class_id
        if not mask.any():
            continue
        class_features = features[mask]
        weights = probabilities[mask, class_id].clamp_min(1e-12)
        count[class_id] = len(class_features)
        mass[class_id] = weights.sum()
        means[class_id] = (weights[:, None] * class_features).sum(0) / mass[class_id]
        normalized_entropy[class_id] = selected_entropy[mask].mean() / log_classes
        residual = class_features - means[class_id]
        scatter += gpu_ops.matmul(residual.T, residual)
    return ClientPacket(
        count=count,
        mass=mass,
        mean=means,
        entropy=normalized_entropy,
        scatter=scatter,
        scatter_dof=max(int(count.sum().item()) - int((count > 0).sum().item()), 1),
    )


def fit_fedpasa(
    banks: tuple[dict[int, PatchBank], ...],
    text_features: torch.Tensor,
    *,
    entropy_strength: float = 2.0,
    consensus_strength: float = 8.0,
    prior_strength: float = 8.0,
    purification_temperature: float = 100.0,
) -> Adapter:
    """Fit reliability-aware GDA using only mergeable client packets.

    The reliability is fully training-side and label-free: low-entropy local evidence and
    agreement with the leave-one-population consensus receive larger precision weight.
    """

    packets = tuple(
        make_client_packet(bank, text_features, purification_temperature=purification_temperature)
        for bank in banks
    )
    return fit_fedpasa_from_packets(
        packets,
        text_features,
        entropy_strength=entropy_strength,
        consensus_strength=consensus_strength,
        prior_strength=prior_strength,
    )


def fit_fedpasa_from_packets(
    packets: tuple[ClientPacket, ...],
    text_features: torch.Tensor,
    *,
    entropy_strength: float = 2.0,
    consensus_strength: float = 8.0,
    prior_strength: float = 8.0,
) -> Adapter:
    """Server-side fit; accepts only the packets defined by :class:`ClientPacket`."""

    if not packets:
        raise ValueError("at least one client packet is required")
    text = F.normalize(text_features.double().cpu(), dim=-1)
    classes, dimension = text.shape
    masses = torch.stack([packet.mass for packet in packets])
    counts = torch.stack([packet.count for packet in packets])
    local_means = F.normalize(torch.stack([packet.mean for packet in packets]), dim=-1)
    entropies = torch.stack([packet.entropy for packet in packets])
    present = counts > 0

    preliminary_weight = masses * present
    preliminary = torch.einsum("kc,kcd->cd", preliminary_weight, local_means)
    preliminary = F.normalize(preliminary + 1e-8 * text, dim=-1)
    disagreement = 1.0 - torch.einsum("kcd,cd->kc", local_means, preliminary).clamp(-1.0, 1.0)
    reliability = torch.exp(-entropy_strength * entropies - consensus_strength * disagreement)
    reliability = reliability * present
    effective = masses * reliability

    observed = torch.einsum("kc,kcd->cd", effective, local_means)
    observed = observed / effective.sum(0).clamp_min(1e-12)[:, None]
    # Empirical-Bayes shrinkage is relevant only for weakly supported classes.
    shrink = effective.sum(0) / (effective.sum(0) + prior_strength)
    means = F.normalize(shrink[:, None] * observed + (1.0 - shrink[:, None]) * text, dim=-1)

    # Client-shared within-class scatter plus between-client mean correction.
    client_reliability = (
        (reliability * counts).sum(1) / counts.sum(1).clamp_min(1.0)
    ).clamp_min(1e-6)
    scatter = torch.zeros((dimension, dimension), dtype=torch.float64)
    dof = 0.0
    for client_id, packet in enumerate(packets):
        r_client = client_reliability[client_id]
        scatter += r_client * packet.scatter
        dof += float(r_client * packet.scatter_dof)
        for class_id in range(classes):
            if packet.count[class_id] <= 0:
                continue
            delta = local_means[client_id, class_id] - means[class_id]
            scatter += (
                reliability[client_id, class_id]
                * packet.count[class_id]
                * torch.outer(delta, delta)
            )
            dof += float(reliability[client_id, class_id] * packet.count[class_id])

    trace_covariance = torch.trace(scatter) / max(dof, 1.0)
    regularized = scatter + trace_covariance * torch.eye(dimension, dtype=torch.float64)
    precision = dimension * gpu_ops.pinv(regularized)
    weight = precision @ means.T
    bias = -0.5 * torch.einsum("cd,de,ce->c", means, precision, means) - np.log(classes)
    return Adapter(
        weight.float(),
        bias.float(),
        {
            "method": "fedpasa_reliability_statistics",
            "clients": len(packets),
            "selected_patches": int(counts.sum().item()),
            "mean_reliability": float(reliability[present].mean().item()),
            "minimum_class_effective_mass": float(effective.sum(0).min().item()),
            "entropy_strength": entropy_strength,
            "consensus_strength": consensus_strength,
            "prior_strength": prior_strength,
        },
    )
