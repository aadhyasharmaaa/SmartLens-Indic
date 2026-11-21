from __future__ import annotations

from collections import OrderedDict
from typing import Any


def group_word_regions(
    word_details: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    padding_ratio: float = 0.18,
) -> list[dict[str, Any]]:
    height, width = image_shape[:2]
    grouped: OrderedDict[tuple[str, str, str], list[dict[str, Any]]] = OrderedDict()
    for index, word in enumerate(word_details):
        key = (
            str(word.get("block_num", "0")),
            str(word.get("par_num", "0")),
            str(word.get("line_num", index)),
        )
        grouped.setdefault(key, []).append(word)

    regions: list[dict[str, Any]] = []
    for key, words in grouped.items():
        left = min(word["left"] for word in words)
        top = min(word["top"] for word in words)
        right = max(word["left"] + word["width"] for word in words)
        bottom = max(word["top"] + word["height"] for word in words)
        horizontal_padding = max(3, round((bottom - top) * padding_ratio))
        vertical_padding = max(3, round((bottom - top) * padding_ratio))
        regions.append(
            {
                "line_key": key,
                "left": max(0, left - horizontal_padding),
                "top": max(0, top - vertical_padding),
                "right": min(width, right + horizontal_padding),
                "bottom": min(height, bottom + vertical_padding),
                "text": " ".join(word["text"] for word in words),
                "confidence": _mean_confidence(words),
                "words": words,
            }
        )

    regions.sort(key=lambda region: (region["top"], region["left"]))
    deduplicated: list[dict[str, Any]] = []
    for region in regions:
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(deduplicated)
                if _intersection_over_union(region, previous) >= 0.7
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(region)
        elif _confidence_value(region) > _confidence_value(deduplicated[duplicate_index]):
            deduplicated[duplicate_index] = region

    deduplicated.sort(key=lambda region: (region["top"], region["left"]))
    return deduplicated


def _mean_confidence(words: list[dict[str, Any]]) -> float | None:
    confidences = [
        float(word["confidence"])
        for word in words
        if word.get("confidence") is not None
    ]
    return sum(confidences) / len(confidences) if confidences else None


def _confidence_value(region: dict[str, Any]) -> float:
    confidence = region.get("confidence")
    return float(confidence) if confidence is not None else -1.0


def _intersection_over_union(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    intersection_width = max(
        0,
        min(first["right"], second["right"]) - max(first["left"], second["left"]),
    )
    intersection_height = max(
        0,
        min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]),
    )
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    first_area = (first["right"] - first["left"]) * (first["bottom"] - first["top"])
    second_area = (second["right"] - second["left"]) * (second["bottom"] - second["top"])
    return intersection / (first_area + second_area - intersection)
