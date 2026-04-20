from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import numpy as np


def text_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(None, left, right).ratio())


def assert_deterministic_output_match(
    left: str,
    right: str,
    *,
    left_label: str,
    right_label: str,
) -> None:
    if left != right:
        similarity = text_similarity(left, right)
        raise AssertionError(
            f"{left_label} != {right_label} under deterministic generation "
            f"(sequence similarity={similarity:.3f})\n"
            f"{left_label}: {left!r}\n"
            f"{right_label}: {right!r}"
        )


def _to_row_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    raise AssertionError(f"Expected 1D or 2D activation data, got shape {array.shape}")


def _row_cosine(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= eps or right_norm <= eps:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def _topk_overlap_fraction(left: np.ndarray, right: np.ndarray, *, k: int) -> float:
    if left.shape != right.shape:
        raise AssertionError(f"Activation shape mismatch: {left.shape} vs {right.shape}")
    k = min(int(k), left.size)
    left_indices = set(np.argsort(np.abs(left))[-k:].tolist())
    right_indices = set(np.argsort(np.abs(right))[-k:].tolist())
    return float(len(left_indices & right_indices) / max(k, 1))


def assert_activation_structure_stable(
    actual_rows: Any,
    reference_rows: Any,
    *,
    min_mean_cosine: float = 0.90,
    min_mean_topk_overlap: float = 0.50,
    top_k: int = 10,
) -> None:
    actual = _to_row_matrix(actual_rows)
    reference = _to_row_matrix(reference_rows)
    if actual.shape != reference.shape:
        raise AssertionError(
            f"Activation shape mismatch: actual {actual.shape} vs reference {reference.shape}"
        )

    if not np.any(np.abs(actual) > 1e-12):
        raise AssertionError("Activation array is entirely zero")

    cosines = []
    overlaps = []
    for actual_row, reference_row in zip(actual, reference, strict=True):
        cosine = _row_cosine(actual_row, reference_row)
        if cosine is not None:
            cosines.append(cosine)
        overlaps.append(_topk_overlap_fraction(actual_row, reference_row, k=top_k))

    mean_cosine = float(np.mean(cosines)) if cosines else None
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    if mean_cosine is None or mean_cosine < float(min_mean_cosine):
        raise AssertionError(
            f"Mean activation cosine below threshold: {mean_cosine} < {min_mean_cosine}"
        )
    if mean_overlap < float(min_mean_topk_overlap):
        raise AssertionError(
            f"Mean top-k overlap below threshold: {mean_overlap} < {min_mean_topk_overlap}"
        )
