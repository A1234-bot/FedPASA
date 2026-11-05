"""Opt-in CUDA dense algebra; CPU packet/tie semantics stay unchanged.

Accuracy workers enable this explicitly. Timing experiments retain their
original backend so their measurements remain comparable.
"""
import os
import torch


def enabled():
    return os.environ.get("FEDPASA_GPU_ACCURACY") == "1"


def matmul(a, b):
    if not enabled():
        return a @ b
    # Preserve FP64 statistics and full FP32 products (no TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    return (a.to("cuda") @ b.to("cuda")).cpu()


def pinv(a):
    # On this GPU the FP64 SVD is slower than the CPU implementation.
    return torch.linalg.pinv(a)


def solve(a, b):
    return torch.linalg.solve(a, b)


def average_precision(targets, scores):
    """CUDA sorting with the original CPU AP accumulation and tie grouping.

    Keeping the final reduction identical avoids changing grid-search choices
    when candidates have exactly tied AP. No CUDA boolean indexing is used.
    """
    import numpy as np
    y = torch.as_tensor(np.asarray(targets), device="cuda", dtype=torch.float32)
    s = torch.as_tensor(np.asarray(scores), device="cuda")
    order = torch.argsort(s, dim=0, descending=True, stable=True)
    ordered_s = torch.gather(s, 0, order).cpu().numpy()
    ordered_y = torch.gather(y, 0, order).cpu().numpy()
    aps = []
    for c in range(scores.shape[1]):
        ends = np.r_[np.flatnonzero(np.diff(ordered_s[:, c])), len(scores) - 1]
        tps = np.cumsum(ordered_y[:, c], dtype=np.float64)[ends]
        fps = 1 + ends - tps
        precision = tps / (tps + fps)
        recall = np.ones_like(tps) if tps[-1] == 0 else tps / tps[-1]
        recall = np.r_[recall[::-1], 0]
        aps.append(-np.sum(np.diff(recall) * precision[::-1]))
    per_class = np.asarray(aps, dtype=np.float64)
    return float(100.0 * per_class.mean()), 100.0 * per_class


def merge_batch(banks, flat, probabilities, entropy, predictions, origins, k, bank_type):
    """CPU top-k retains tie order; CUDA gathers only selected new rows.

    No CUDA boolean indexing. One packed integer gather per batch avoids
    transferring and masking every feature on CPU for every class.
    """
    entropy, predictions = entropy.cpu(), predictions.cpu()
    plans, packed = [], []
    for class_id in predictions.unique().tolist():
        rows = torch.nonzero(predictions == class_id, as_tuple=False).flatten()
        previous = banks.get(class_id)
        nold = 0 if previous is None else len(previous.entropy)
        scores = entropy[rows]
        if previous is not None:
            scores = torch.cat((previous.entropy, scores))
        keep = torch.topk(scores, min(k, len(scores)), largest=False, sorted=False).indices
        new_positions = torch.nonzero(keep >= nold, as_tuple=False).flatten()
        new_rows = rows[keep[new_positions] - nold]
        packed.append(new_rows)
        plans.append((class_id, previous, keep, new_positions, new_rows, scores[keep], nold))
    all_rows = torch.cat(packed).to(flat.device)
    selected_x = torch.index_select(flat, 0, all_rows).cpu()
    selected_p = torch.index_select(probabilities, 0, all_rows).cpu()
    offset = 0
    for class_id, previous, keep, new_pos, new_rows, scores, nold in plans:
        x = torch.empty((len(keep), flat.shape[1]), dtype=flat.dtype)
        p = torch.empty((len(keep), probabilities.shape[1]), dtype=probabilities.dtype)
        source = torch.empty(len(keep), dtype=torch.long)
        old_pos = torch.nonzero(keep < nold, as_tuple=False).flatten()
        if len(old_pos):
            old_rows = keep[old_pos]
            x[old_pos] = previous.features[old_rows]
            p[old_pos] = previous.probabilities[old_rows]
            source[old_pos] = previous.image_indices[old_rows]
        length = len(new_pos)
        x[new_pos] = selected_x[offset:offset + length]
        p[new_pos] = selected_p[offset:offset + length]
        source[new_pos] = origins[new_rows]
        offset += length
        banks[class_id] = bank_type(x, p, scores, source)
