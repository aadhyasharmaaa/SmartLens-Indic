from __future__ import annotations

import numpy as np
import pytest

from recognition.paddle_devanagari_recognizer import PaddleDevanagariRecognizer


class FakePrediction:
    json = {"res": {"rec_text": "स्पोर्ट", "rec_score": 0.96}}


class FakeModel:
    def predict(self, *, input):
        assert input.shape == (32, 80, 3)
        return [FakePrediction()]


def test_paddle_recognizer_uses_local_model_and_returns_score(tmp_path) -> None:
    (tmp_path / "inference.yml").write_text("model_name: test\n", encoding="utf-8")
    recognizer = PaddleDevanagariRecognizer(
        model_dir=tmp_path,
        device="cpu",
        model=FakeModel(),
    )

    result = recognizer.recognize(np.zeros((32, 80, 3), dtype=np.uint8))

    assert result["text"] == "स्पोर्ट"
    assert result["confidence"] == pytest.approx(0.96)
    assert "not calibrated probability" in result["confidence_type"]
    assert recognizer.device == "cpu"


def test_paddle_recognizer_rejects_missing_local_model(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="recognition model"):
        PaddleDevanagariRecognizer(model_dir=tmp_path, model=FakeModel())


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((32, 80), dtype=np.uint8),
        np.zeros((32, 80, 3), dtype=np.float32),
        np.zeros((32, 80, 4), dtype=np.uint8),
    ],
)
def test_paddle_recognizer_rejects_invalid_input(image, tmp_path) -> None:
    (tmp_path / "inference.yml").write_text("model_name: test\n", encoding="utf-8")
    recognizer = PaddleDevanagariRecognizer(
        model_dir=tmp_path,
        device="cpu",
        model=FakeModel(),
    )

    with pytest.raises(ValueError):
        recognizer.recognize(image)
