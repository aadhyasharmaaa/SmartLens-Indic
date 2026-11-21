from __future__ import annotations

import numpy as np

from recognition.paddle_text_detector import PaddleTextDetector


def test_scene_detector_regions_do_not_merge_through_a_vertical_chain() -> None:
    # Axis-aligned bounds observed from the local detector on scene image_100.jpg.
    scene_image_100_boxes = [
        (1740, 1091, 2045, 1186),
        (2081, 1076, 2605, 1209),
        (2615, 1059, 3007, 1222),
        (2934, 960, 3238, 1151),
        (423, 853, 1679, 1252),
        (1955, 774, 2235, 917),
        (2230, 760, 2656, 934),
        (1023, 723, 1120, 758),
        (498, 456, 1655, 841),
        (1591, 431, 2007, 873),
        (1931, 393, 2732, 792),
        (2837, 412, 3265, 706),
        (445, 264, 1678, 623),
        (2623, 259, 2832, 410),
        (1977, 218, 2669, 448),
    ]

    class FakeModel:
        def predict(self, image, batch_size):
            assert batch_size == 1
            polygons = [
                [
                    [left, top],
                    [right - 1, top],
                    [right - 1, bottom - 1],
                    [left, bottom - 1],
                ]
                for left, top, right, bottom in scene_image_100_boxes
            ]
            prediction = type(
                "FakePrediction",
                (),
                {
                    "json": {
                        "res": {
                            "dt_polys": polygons,
                            "dt_scores": [0.8] * len(polygons),
                        }
                    }
                },
            )()
            return [prediction]

    detector = object.__new__(PaddleTextDetector)
    detector._model = FakeModel()

    regions = detector.detect(np.zeros((1520, 3264, 3), dtype=np.uint8))

    assert len(regions) == 10
    assert not any(
        (region["left"], region["top"], region["right"], region["bottom"])
        == (445, 218, 3264, 873)
        for region in regions
    )
    assert max(
        len(region.get("polygons", [region.get("polygon")]))
        for region in regions
    ) <= 2
