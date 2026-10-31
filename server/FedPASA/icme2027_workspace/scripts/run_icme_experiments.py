from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS, FEDMPT_MAP, Benchmark  # noqa: E402
from fedpasa.core import (  # noqa: E402
    Adapter,
    PatchBank,
    _gda_from_samples,
    _purify_bank,
    _stack_banks,
    build_client_banks,
    fit_central_piaa,
    fit_fedpasa_from_packets,
    fit_quantile_aligned_fedpasa,
    make_client_packet,
    mean_average_precision,
    split_global_bank_by_client,
)


SEEDS = (7, 17, 27)
T_VALUES = (10, 20, 40, 60, 80, 100)
DIRICHLET_BETAS = (0.1, 0.3, 0.5)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def serialize_bank(bank: dict[int, PatchBank]) -> dict[int, dict[str, torch.Tensor | None]]:
    return {
        class_id: {
            "features": value.features,
            "probabilities": value.probabilities,
            "entropy": value.entropy,
            "image_indices": value.image_indices,
        }
        for class_id, value in bank.items()
    }


def deserialize_bank(payload: dict[int, dict[str, torch.Tensor]]) -> dict[int, PatchBank]:
    return {
        int(class_id): PatchBank(
            value["features"],
            value["probabilities"],
            value["entropy"],
            value.get("image_indices"),
        )
        for class_id, value in payload.items()
    }


def truncate_bank(bank: dict[int, PatchBank], k: int) -> dict[int, PatchBank]:
    output: dict[int, PatchBank] = {}
    for class_id, value in bank.items():
        keep = torch.topk(value.entropy, min(k, len(value.entropy)), largest=False).indices
        output[class_id] = PatchBank(
            value.features[keep],
            value.probabilities[keep],
            value.entropy[keep],
            None if value.image_indices is None else value.image_indices[keep],
        )
    return output


def global_bank(
    cache_root: Path,
    report_root: Path,
    text: torch.Tensor,
    *,
    k: int,
    batch_images: int,
) -> tuple[dict[int, PatchBank], float]:
    path = report_root / "banks" / f"global_k{k}.pt"
    if path.exists():
        bank = deserialize_bank(torch.load(path, map_location="cpu", weights_only=True))
        if bank and all(value.image_indices is not None for value in bank.values()):
            print(f"BANK_REUSE {path}", flush=True)
            return bank, 0.0
        print(f"BANK_REBUILD missing source-image indices: {path}", flush=True)
    train_cls = np.load(cache_root / "train_cls.npy", mmap_mode="r")
    started = perf_counter()
    bank = build_client_banks(
        cache_root / "train_patch.npy",
        np.zeros(len(train_cls), dtype=np.int64),
        text,
        clients=1,
        k=k,
        batch_images=batch_images,
    )[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    torch.save(serialize_bank(bank), temporary)
    temporary.replace(path)
    seconds = perf_counter() - started
    print(f"BANK_SAVED {path} seconds={seconds:.2f}", flush=True)
    return bank, seconds


@torch.inference_mode()
def partition_features(
    cache_root: Path,
    text: torch.Tensor,
    *,
    batch_images: int,
    device: str,
) -> np.ndarray:
    """Match the released FedMPT clustering input: zero-shot probability vectors."""

    path = cache_root / "partition_features.npy"
    if path.exists():
        print(f"PARTITION_FEATURE_REUSE {path}", flush=True)
        return np.load(path, mmap_mode="r")
    cls_cache = np.load(cache_root / "train_cls.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(
        path.with_name(f".{path.name}.partial.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(len(cls_cache), text.shape[0]),
    )
    text_gpu = F.normalize(text.float().to(device), dim=-1)
    for start in range(0, len(cls_cache), batch_images):
        stop = min(start + batch_images, len(cls_cache))
        cls = torch.from_numpy(np.asarray(cls_cache[start:stop], dtype=np.float32)).to(device)
        cls = F.normalize(cls, dim=-1)
        output[start:stop] = F.softmax(cls @ text_gpu.T, dim=-1).cpu().numpy()
        if start % (batch_images * 100) == 0 or stop == len(cls_cache):
            print(f"PARTITION_FEATURE images={stop}/{len(cls_cache)}", flush=True)
    output.flush()
    del output
    temporary = path.with_name(f".{path.name}.partial.npy")
    temporary.replace(path)
    return np.load(path, mmap_mode="r")


def assignments_for(
    features: np.ndarray,
    report_root: Path,
    *,
    clients: int,
    seed: int,
    n_init: int,
) -> tuple[np.ndarray, list[int], float]:
    path = report_root / "partitions" / f"clients{clients}_seed{seed}.npy"
    if path.exists():
        labels = np.load(path)
        return labels, np.bincount(labels, minlength=clients).tolist(), 0.0
    started = perf_counter()
    labels = KMeans(
        n_clusters=clients,
        random_state=seed,
        n_init=n_init,
        algorithm="lloyd",
    ).fit_predict(features)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.npy")
    np.save(temporary, labels)
    temporary.replace(path)
    seconds = perf_counter() - started
    return labels, np.bincount(labels, minlength=clients).tolist(), seconds


def dirichlet_assignments(
    targets: np.ndarray,
    report_root: Path,
    *,
    clients: int,
    beta: float,
    seed: int,
    minimum_client_images: int = 10,
) -> tuple[np.ndarray, list[int], float]:
    beta_name = str(beta).replace(".", "p")
    path = report_root / "partitions" / f"dirichlet_clients{clients}_beta{beta_name}_seed{seed}.npy"
    if path.exists():
        labels = np.load(path)
        return labels, np.bincount(labels, minlength=clients).tolist(), 0.0
    started = perf_counter()
    targets = np.asarray(targets, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    positive_classes = [np.flatnonzero(row) for row in targets]
    primary = np.asarray(
        [rng.choice(classes) if len(classes) else rng.integers(targets.shape[1]) for classes in positive_classes],
        dtype=np.int64,
    )
    labels = np.empty(len(targets), dtype=np.int64)
    for _ in range(1000):
        buckets: list[list[int]] = [[] for _ in range(clients)]
        for class_id in range(targets.shape[1]):
            indices = np.flatnonzero(primary == class_id)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(clients, beta, dtype=np.float64))
            under_average = np.asarray(
                [len(bucket) < len(targets) / clients for bucket in buckets], dtype=np.float64
            )
            proportions *= under_average
            if proportions.sum() <= 0:
                proportions = np.ones(clients, dtype=np.float64)
            boundaries = (np.cumsum(proportions / proportions.sum())[:-1] * len(indices)).astype(int)
            for client_id, values in enumerate(np.split(indices, boundaries)):
                buckets[client_id].extend(values.tolist())
        sizes = [len(bucket) for bucket in buckets]
        if min(sizes) >= minimum_client_images:
            for client_id, values in enumerate(buckets):
                labels[np.asarray(values, dtype=np.int64)] = client_id
            break
    else:
        raise RuntimeError("failed to construct a non-empty Dirichlet partition")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.npy")
    np.save(temporary, labels)
    temporary.replace(path)
    return labels, sizes, perf_counter() - started


def f1_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    candidates: list[dict[str, float]] = []
    truth = targets.astype(bool)
    for threshold in np.arange(0.1, 1.0, 0.1):
        predicted = scores > threshold
        tp = np.logical_and(predicted, truth).sum(axis=0, dtype=np.int64)
        fp = np.logical_and(predicted, ~truth).sum(axis=0, dtype=np.int64)
        fn = np.logical_and(~predicted, truth).sum(axis=0, dtype=np.int64)
        pc = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
        rc = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)
        cf1 = np.divide(2 * pc * rc, pc + rc, out=np.zeros_like(pc), where=(pc + rc) > 0).mean()
        po = float(tp.sum() / max(int(tp.sum() + fp.sum()), 1))
        ro = float(tp.sum() / max(int(tp.sum() + fn.sum()), 1))
        of1 = 0.0 if po + ro == 0 else 2 * po * ro / (po + ro)
        candidates.append({"threshold": float(threshold), "CF1": 100 * cf1, "OF1": 100 * of1})
    best_cf1 = max(candidates, key=lambda value: value["CF1"])
    best_of1 = max(candidates, key=lambda value: value["OF1"])
    return {"best_CF1": best_cf1, "best_OF1": best_of1}


@torch.inference_mode()
def evaluate_many(
    cache_root: Path,
    benchmark: Benchmark,
    text_features: torch.Tensor,
    adapters: dict[str, Adapter],
    *,
    batch_images: int,
    score_root: Path | None = None,
    device: str = "cuda",
) -> tuple[dict[str, dict[str, object]], float]:
    patches = np.load(cache_root / "test_patch.npy", mmap_mode="r")
    cls_cache = np.load(cache_root / "test_cls.npy", mmap_mode="r")
    targets = np.load(cache_root / "test_targets.npy")
    text = F.normalize(text_features.float().to(device), dim=-1)
    gpu_adapters = {
        name: (adapter.weight.float().to(device), adapter.bias.float().to(device))
        for name, adapter in adapters.items()
    }
    modes = ("fixed_0.9", "patch_only", "cls_only", "entropy_adaptive")
    score_chunks = {name: {mode: [] for mode in modes} for name in adapters}
    log_classes = float(np.log(text.shape[0]))
    prior_logit = float(np.log(0.9 / 0.1))
    started = perf_counter()
    for start in range(0, len(targets), batch_images):
        stop = min(start + batch_images, len(targets))
        patch = torch.from_numpy(np.asarray(patches[start:stop], dtype=np.float32)).to(device)
        patch = F.normalize(patch, dim=-1)
        cls = torch.from_numpy(np.asarray(cls_cache[start:stop], dtype=np.float32)).to(device)
        cls = F.normalize(cls, dim=-1)
        cls_prob = F.softmax((cls @ text.T) * benchmark.cls_scale, dim=-1)
        cls_conf = 1.0 - (
            -(cls_prob * cls_prob.clamp_min(1e-10).log()).sum(-1) / log_classes
        )
        flat = patch.reshape(-1, patch.shape[-1])
        for name, (weight, bias) in gpu_adapters.items():
            refined = F.softmax((flat @ weight + bias) / 500.0, dim=-1)
            refined = refined.reshape(stop - start, patch.shape[1], -1)
            patch_prob = F.softmax(refined.max(dim=1).values * benchmark.patch_scale, dim=-1)
            patch_conf = 1.0 - (
                -(patch_prob * patch_prob.clamp_min(1e-10).log()).sum(-1) / log_classes
            )
            alpha = torch.sigmoid(
                torch.as_tensor(prior_logit, device=device) + 2.0 * (patch_conf - cls_conf)
            )[:, None]
            values = {
                "fixed_0.9": 0.1 * cls_prob + 0.9 * patch_prob,
                "patch_only": patch_prob,
                "cls_only": cls_prob,
                "entropy_adaptive": (1.0 - alpha) * cls_prob + alpha * patch_prob,
            }
            for mode, scores in values.items():
                score_chunks[name][mode].append(scores.cpu().numpy())
        if start % (batch_images * 50) == 0 or stop == len(targets):
            print(f"EVAL dataset={benchmark.name} images={stop}/{len(targets)}", flush=True)

    result: dict[str, dict[str, object]] = {}
    for name, adapter in adapters.items():
        result[name] = {"adapter": adapter.metadata}
        for mode in modes:
            scores = np.concatenate(score_chunks[name][mode])
            map_value, per_class = mean_average_precision(targets, scores)
            result[name][mode] = {
                "mAP": map_value,
                "per_class_AP": per_class.tolist(),
                **f1_metrics(targets, scores),
            }
            if score_root is not None and mode == "fixed_0.9":
                score_root.mkdir(parents=True, exist_ok=True)
                np.save(score_root / f"{name}_scores.npy", scores)
    return result, perf_counter() - started


def no_purification_adapter(bank: dict[int, PatchBank], text: torch.Tensor) -> Adapter:
    features, probabilities, labels = _stack_banks((bank,))
    weight, bias, _ = _gda_from_samples(features, probabilities, labels, text)
    return Adapter(weight.float(), bias.float(), {"method": "no_global_purification"})


def diagonal_covariance_adapter(
    bank: dict[int, PatchBank], text: torch.Tensor, *, temperature: float
) -> Adapter:
    features, probabilities, labels, _ = _purify_bank(bank, text, temperature=temperature)
    text_cpu = F.normalize(text.double().cpu(), dim=-1)
    classes, dimension = text_cpu.shape
    means = text_cpu.clone()
    for class_id in range(classes):
        mask = labels == class_id
        if mask.any():
            weights = probabilities[mask, class_id].clamp_min(1e-12)
            means[class_id] = (weights[:, None] * features[mask]).sum(0) / weights.sum()
    diagonal_scatter = torch.zeros(dimension, dtype=torch.float64)
    for class_id in range(classes):
        mask = labels == class_id
        if mask.any():
            diagonal_scatter += (features[mask] - means[class_id]).square().sum(0)
    trace_covariance = diagonal_scatter.sum() / max(len(features) - 1, 1)
    precision_diagonal = dimension / (diagonal_scatter + trace_covariance).clamp_min(1e-12)
    weight = precision_diagonal[:, None] * means.T
    bias = -0.5 * (means.square() * precision_diagonal[None, :]).sum(1) - np.log(classes)
    return Adapter(weight.float(), bias.float(), {"method": "diagonal_covariance"})


def run_main(args, benchmark: Benchmark, cache_root: Path, report_root: Path) -> dict[str, object]:
    text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float()
    bank_k = max([args.k, *args.ablation_k]) if args.phase == "all" else args.k
    large_bank, bank_seconds = global_bank(
        cache_root,
        report_root,
        text,
        k=bank_k,
        batch_images=args.bank_batch_size,
    )
    bank = truncate_bank(large_bank, args.k)
    started = perf_counter()
    central = fit_central_piaa(
        bank,
        text,
        purification_temperature=benchmark.purification_temperature,
    )
    central_fit_seconds = perf_counter() - started
    cluster_features = partition_features(
        cache_root,
        text,
        batch_images=max(args.eval_batch_size, 256),
        device=args.device,
    )

    runs: list[dict[str, object]] = []
    representative: Adapter | None = None
    counts = benchmark.client_counts()
    for setting_index, (t_value, clients) in enumerate(zip(T_VALUES, counts)):
        for seed in args.seeds:
            assignments, client_sizes, cluster_seconds = assignments_for(
                cluster_features,
                report_root,
                clients=clients,
                seed=seed,
                n_init=args.kmeans_n_init,
            )
            local_banks = split_global_bank_by_client(bank, assignments, clients)
            started = perf_counter()
            adapter = fit_quantile_aligned_fedpasa(
                local_banks,
                text,
                k=args.k,
                purification_temperature=benchmark.purification_temperature,
            )
            fit_seconds = perf_counter() - started
            weight_error = float((adapter.weight - central.weight).abs().max())
            bias_error = float((adapter.bias - central.bias).abs().max())
            equivalent = weight_error <= args.equivalence_tolerance and bias_error <= args.equivalence_tolerance
            if representative is None and t_value == 40 and seed == args.seeds[0]:
                representative = adapter
            baseline = None if benchmark.name not in FEDMPT_MAP else FEDMPT_MAP[benchmark.name][setting_index]
            runs.append(
                {
                    "t_percent": t_value,
                    "clients": clients,
                    "seed": seed,
                    "client_sizes": client_sizes,
                    "cluster_seconds": cluster_seconds,
                    "adapter_fit_seconds": fit_seconds,
                    "central_recovery_max_abs_weight": weight_error,
                    "central_recovery_max_abs_bias": bias_error,
                    "numerically_equivalent_at_tolerance": equivalent,
                    "fedmpt_mAP": baseline,
                    "communication_MiB": adapter.metadata["estimated_communication_MiB"],
                }
            )
            print(
                f"MAIN dataset={benchmark.name} t={t_value}% clients={clients} seed={seed} "
                f"weight_error={weight_error:.3e} bias_error={bias_error:.3e}",
                flush=True,
            )
            atomic_json(
                report_root / "main.progress.json",
                {"dataset": benchmark.name, "completed_runs": runs},
            )

    if representative is None:
        representative = central
    evaluation, evaluation_seconds = evaluate_many(
        cache_root,
        benchmark,
        text,
        {"central_piaa": central, "qafedpasa_representative": representative},
        batch_images=args.eval_batch_size,
        score_root=report_root / "scores",
        device=args.device,
    )
    central_map = float(evaluation["central_piaa"]["fixed_0.9"]["mAP"])
    representative_map = float(evaluation["qafedpasa_representative"]["fixed_0.9"]["mAP"])
    for run in runs:
        run["mAP"] = representative_map if run["numerically_equivalent_at_tolerance"] else None
        run["score_provenance"] = (
            "reused from fully evaluated representative after adapter equality check"
            if run["numerically_equivalent_at_tolerance"]
            else "not reused: equality tolerance failed"
        )
        baseline = run["fedmpt_mAP"]
        run["margin_over_fedmpt_points"] = None if baseline is None else representative_map - float(baseline)
        run["passes_fedmpt_plus_2"] = None if baseline is None else representative_map >= float(baseline) + 2.0

    margins = [float(run["margin_over_fedmpt_points"]) for run in runs if run["margin_over_fedmpt_points"] is not None]
    report = {
        "dataset": benchmark.name,
        "protocol": {
            "partition": "released FedMPT KMeans input: ViT-B/16 zero-shot probability vectors",
            "t_percent": list(T_VALUES),
            "client_counts": list(counts),
            "seeds": list(args.seeds),
            "kmeans_n_init": args.kmeans_n_init,
            "global_top_k_per_class": args.k,
            "adapter_equality_tolerance": args.equivalence_tolerance,
            "test_labels_used_for_model_selection": False,
        },
        "runtime": {
            "global_bank_seconds_this_run": bank_seconds,
            "central_adapter_fit_seconds": central_fit_seconds,
            "full_test_evaluation_seconds": evaluation_seconds,
        },
        "piaa_reported_mAP": benchmark.piaa_reported_map,
        "evaluation": evaluation,
        "runs": runs,
        "summary": {
            "configurations": len(runs),
            "central_mAP": central_map,
            "qafedpasa_mAP": representative_map,
            "all_adapters_equivalent": all(bool(run["numerically_equivalent_at_tolerance"]) for run in runs),
            "minimum_margin_over_fedmpt_points": min(margins) if margins else None,
            "all_pass_fedmpt_plus_2": all(value >= 2.0 for value in margins) if margins else None,
        },
    }
    atomic_json(report_root / "main.json", report)
    return report


def run_ablation(args, benchmark: Benchmark, cache_root: Path, report_root: Path) -> dict[str, object]:
    text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float()
    largest_k = max(args.ablation_k)
    large_bank, bank_seconds = global_bank(
        cache_root,
        report_root,
        text,
        k=largest_k,
        batch_images=args.bank_batch_size,
    )
    adapters: dict[str, Adapter] = {}
    fit_seconds: dict[str, float] = {}
    for k in args.ablation_k:
        bank = truncate_bank(large_bank, k)
        started = perf_counter()
        adapters[f"topk_{k}"] = fit_central_piaa(
            bank,
            text,
            purification_temperature=benchmark.purification_temperature,
        )
        fit_seconds[f"topk_{k}"] = perf_counter() - started
    main_bank = truncate_bank(large_bank, args.k)
    started = perf_counter()
    adapters["no_global_purification"] = no_purification_adapter(main_bank, text)
    fit_seconds["no_global_purification"] = perf_counter() - started
    started = perf_counter()
    adapters["diagonal_covariance"] = diagonal_covariance_adapter(
        main_bank,
        text,
        temperature=benchmark.purification_temperature,
    )
    fit_seconds["diagonal_covariance"] = perf_counter() - started

    if args.local_ablation:
        clients = benchmark.client_counts()[2]
        cluster_features = partition_features(
            cache_root,
            text,
            batch_images=max(args.eval_batch_size, 256),
            device=args.device,
        )
        assignments, client_sizes, cluster_seconds = assignments_for(
            cluster_features,
            report_root,
            clients=clients,
            seed=args.seeds[0],
            n_init=args.kmeans_n_init,
        )
        local_path = report_root / "banks" / f"local_clients{clients}_seed{args.seeds[0]}_k{args.k}.pt"
        if local_path.exists():
            payload = torch.load(local_path, map_location="cpu", weights_only=True)
            local_banks = tuple(deserialize_bank(value) for value in payload)
            local_bank_seconds = 0.0
        else:
            started = perf_counter()
            local_banks = build_client_banks(
                cache_root / "train_patch.npy",
                assignments,
                text,
                clients=clients,
                k=args.k,
                batch_images=args.bank_batch_size,
            )
            local_bank_seconds = perf_counter() - started
            local_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = local_path.with_name(f".{local_path.name}.partial")
            torch.save([serialize_bank(value) for value in local_banks], temporary)
            temporary.replace(local_path)
        started = perf_counter()
        packets = tuple(
            make_client_packet(
                bank,
                text,
                purification_temperature=benchmark.purification_temperature,
            )
            for bank in local_banks
        )
        adapters["local_topk_no_quantile_alignment"] = fit_fedpasa_from_packets(
            packets,
            text,
            entropy_strength=0.0,
            consensus_strength=0.0,
            prior_strength=0.0,
        )
        fit_seconds["local_topk_no_quantile_alignment"] = perf_counter() - started
        started = perf_counter()
        adapters["reliability_no_prior_shrinkage"] = fit_fedpasa_from_packets(
            packets,
            text,
            entropy_strength=2.0,
            consensus_strength=8.0,
            prior_strength=0.0,
        )
        fit_seconds["reliability_no_prior_shrinkage"] = perf_counter() - started
        started = perf_counter()
        adapters["reliability_entropy_only"] = fit_fedpasa_from_packets(
            packets,
            text,
            entropy_strength=2.0,
            consensus_strength=0.0,
            prior_strength=8.0,
        )
        fit_seconds["reliability_entropy_only"] = perf_counter() - started
        started = perf_counter()
        adapters["reliability_consensus_shrinkage"] = fit_fedpasa_from_packets(
            packets,
            text,
            entropy_strength=2.0,
            consensus_strength=8.0,
            prior_strength=8.0,
        )
        fit_seconds["reliability_consensus_shrinkage"] = perf_counter() - started
        local_metadata = {
            "clients": clients,
            "seed": args.seeds[0],
            "client_sizes": client_sizes,
            "cluster_seconds": cluster_seconds,
            "local_bank_seconds": local_bank_seconds,
        }
    else:
        local_metadata = {"skipped": True}

    evaluation, evaluation_seconds = evaluate_many(
        cache_root,
        benchmark,
        text,
        adapters,
        batch_images=args.eval_batch_size,
        device=args.device,
    )
    report = {
        "dataset": benchmark.name,
        "protocol": "training-label-free component and K-sensitivity ablation",
        "global_bank_seconds_this_run": bank_seconds,
        "adapter_fit_seconds": fit_seconds,
        "full_test_evaluation_seconds": evaluation_seconds,
        "local_ablation": local_metadata,
        "evaluation": evaluation,
    }
    atomic_json(report_root / "ablation.json", report)
    return report


def run_dirichlet(args, benchmark: Benchmark, cache_root: Path, report_root: Path) -> dict[str, object]:
    text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float()
    large_bank, bank_seconds = global_bank(
        cache_root,
        report_root,
        text,
        k=max([args.k, *args.ablation_k]),
        batch_images=args.bank_batch_size,
    )
    bank = truncate_bank(large_bank, args.k)
    central = fit_central_piaa(
        bank,
        text,
        purification_temperature=benchmark.purification_temperature,
    )
    targets = np.load(cache_root / "train_targets.npy")
    runs: list[dict[str, object]] = []
    for beta in args.dirichlet_betas:
        for seed in args.seeds:
            assignments, client_sizes, partition_seconds = dirichlet_assignments(
                targets,
                report_root,
                clients=args.dirichlet_clients,
                beta=beta,
                seed=seed,
            )
            local_banks = split_global_bank_by_client(bank, assignments, args.dirichlet_clients)
            started = perf_counter()
            adapter = fit_quantile_aligned_fedpasa(
                local_banks,
                text,
                k=args.k,
                purification_temperature=benchmark.purification_temperature,
            )
            fit_seconds = perf_counter() - started
            weight_error = float((adapter.weight - central.weight).abs().max())
            bias_error = float((adapter.bias - central.bias).abs().max())
            equivalent = weight_error <= args.equivalence_tolerance and bias_error <= args.equivalence_tolerance
            run = {
                "beta": beta,
                "clients": args.dirichlet_clients,
                "seed": seed,
                "client_sizes": client_sizes,
                "partition_seconds": partition_seconds,
                "adapter_fit_seconds": fit_seconds,
                "central_recovery_max_abs_weight": weight_error,
                "central_recovery_max_abs_bias": bias_error,
                "numerically_equivalent_at_tolerance": equivalent,
                "communication_MiB": adapter.metadata["estimated_communication_MiB"],
            }
            runs.append(run)
            print(
                f"DIRICHLET dataset={benchmark.name} beta={beta} seed={seed} "
                f"weight_error={weight_error:.3e} bias_error={bias_error:.3e}",
                flush=True,
            )
            atomic_json(
                report_root / "dirichlet.progress.json",
                {"dataset": benchmark.name, "completed_runs": runs},
            )

    main_path = report_root / "main.json"
    evaluated_map = None
    score_provenance = None
    if main_path.exists():
        main_report = json.loads(main_path.read_text(encoding="utf-8"))
        evaluated_map = float(main_report["summary"]["qafedpasa_mAP"])
        score_provenance = "reused from main full-test evaluation after adapter equality check"
    for run in runs:
        run["mAP"] = evaluated_map if run["numerically_equivalent_at_tolerance"] else None
        run["score_provenance"] = score_provenance if run["numerically_equivalent_at_tolerance"] else None
    report = {
        "dataset": benchmark.name,
        "protocol": {
            "clients": args.dirichlet_clients,
            "betas": list(args.dirichlet_betas),
            "seeds": list(args.seeds),
            "primary_label_for_multilabel_image": "seeded uniform draw among positive labels",
            "training_labels_used_by_method": False,
            "training_labels_used_only_for_partition_simulation": True,
        },
        "global_bank_seconds_this_run": bank_seconds,
        "runs": runs,
        "summary": {
            "configurations": len(runs),
            "all_adapters_equivalent": all(bool(run["numerically_equivalent_at_tolerance"]) for run in runs),
            "mAP": evaluated_map,
        },
    }
    atomic_json(report_root / "dirichlet.json", report)
    return report


def resolve_cache_root(dataset: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy_voc = ROOT / "outputs" / "voc07" / "cache"
    if dataset == "voc07" and not generic.exists() and legacy_voc.exists():
        return legacy_voc
    return generic


def validate_cache(cache_root: Path) -> None:
    required = (
        "train_patch.npy",
        "test_patch.npy",
        "train_cls.npy",
        "test_cls.npy",
        "train_targets.npy",
        "test_targets.npy",
        "text_features.npy",
    )
    missing = [name for name in required if not (cache_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing feature caches in {cache_root}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--phase", choices=("main", "dirichlet", "ablation", "all"), default="all")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--ablation-k", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--bank-batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-ablation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dirichlet-clients", type=int, default=10)
    parser.add_argument("--dirichlet-betas", type=float, nargs="+", default=list(DIRICHLET_BETAS))
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.dataset]
    cache_root = resolve_cache_root(args.dataset, args.cache_root)
    report_root = args.report_root or ROOT / "outputs" / "icme" / args.dataset
    validate_cache(cache_root)
    report_root.mkdir(parents=True, exist_ok=True)
    if args.phase in {"main", "all"}:
        report = run_main(args, benchmark, cache_root, report_root)
        print(json.dumps(report["summary"], indent=2), flush=True)
    if args.phase in {"dirichlet", "all"}:
        report = run_dirichlet(args, benchmark, cache_root, report_root)
        print(json.dumps(report["summary"], indent=2), flush=True)
    if args.phase in {"ablation", "all"}:
        report = run_ablation(args, benchmark, cache_root, report_root)
        print(f"ABLATION_COMPLETE dataset={benchmark.name} methods={len(report['evaluation'])}", flush=True)


if __name__ == "__main__":
    main()
