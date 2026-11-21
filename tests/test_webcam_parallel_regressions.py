from __future__ import annotations

import cv2
import numpy as np
import pytest

import app
from database.repository import (
    get_recent_readings,
    initialize_database,
    record_reading,
)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _UploadedImage:
    def __init__(self, data: bytes):
        self.data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self.data


class _StreamlitStub:
    def __init__(self, camera_image: _UploadedImage | None = None):
        self.camera_image = camera_image
        self.session_state = {}
        self.info_messages: list[str] = []
        self.text_values: list[str] = []

    def tabs(self, labels):
        return [_Context() for _ in labels]

    def expander(self, *_args, **_kwargs):
        return _Context()

    def selectbox(self, _label, options, **_kwargs):
        return options[0]

    def camera_input(self, *_args, **_kwargs):
        return self.camera_image

    def file_uploader(self, *_args, **_kwargs):
        return None

    def button(self, *_args, **_kwargs):
        return False

    def info(self, message):
        self.info_messages.append(message)

    def text_area(self, _label, *, value, **_kwargs):
        self.text_values.append(value)

    def __getattr__(self, name):
        return lambda *_args, **_kwargs: None


def _encoded_image(color: tuple[int, int, int]) -> _UploadedImage:
    success, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), color, dtype=np.uint8))
    assert success
    return _UploadedImage(encoded.tobytes())


def test_camera_waiting_state_explains_permissions_and_upload_fallback(monkeypatch) -> None:
    streamlit = _StreamlitStub()
    monkeypatch.setattr(app, "st", streamlit)

    app._render_read_page()

    assert any("browser's site permissions" in message for message in streamlit.info_messages)
    assert any("Upload image tab" in message for message in streamlit.info_messages)


def test_camera_capture_is_processed_once_across_reruns(monkeypatch) -> None:
    streamlit = _StreamlitStub(_encoded_image((0, 0, 255)))
    processed: list[np.ndarray] = []
    monkeypatch.setattr(app, "st", streamlit)
    monkeypatch.setattr(app, "_run_reading", lambda image, *_args, **_kwargs: processed.append(image))

    app._render_read_page()
    app._render_read_page()
    assert len(processed) == 1

    streamlit.camera_image = _encoded_image((0, 255, 0))
    app._render_read_page()
    assert len(processed) == 2


def test_oversized_upload_is_rejected_before_copying_its_contents() -> None:
    class OversizedUpload:
        size = app.MAX_IMAGE_BYTES + 1

        def getvalue(self):
            pytest.fail("Oversized upload contents should not be copied")

    with pytest.raises(ValueError, match="upload limit"):
        app._uploaded_image_bytes(OversizedUpload())


def test_history_page_still_renders_saved_readings(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    record_reading(
        {
            "id": "existing-reading",
            "status": "success",
            "input_kind": "upload",
            "recognized_text": "नमस्ते",
            "total_ms": 12.0,
            "ocr_ms": 8.0,
        },
        database_path=database_path,
    )
    monkeypatch.setattr(
        app,
        "get_recent_readings",
        lambda limit: get_recent_readings(limit, database_path=database_path),
    )
    streamlit = _StreamlitStub()
    monkeypatch.setattr(app, "st", streamlit)

    app._render_history()

    assert streamlit.text_values == ["नमस्ते"]
