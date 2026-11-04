from __future__ import annotations


def order_by_reading_boxes(boxes):
    return sorted(boxes, key=lambda box: (box[1], box[0]))
