from __future__ import annotations


def filter_by_confidence(results, threshold: float = 0.5):
    filtered = []
    for item in results:
        confidence = float(item.get("confidence", 0.0))
        if confidence >= threshold:
            filtered.append(item)
    return filtered
