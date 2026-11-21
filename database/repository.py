from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import DATABASE_PATH


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reading_sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    input_kind TEXT NOT NULL,
    original_image_path TEXT,
    preprocessed_image_path TEXT,
    recognized_text TEXT,
    recognition_engine TEXT,
    confidence_mean REAL,
    preprocess_ms REAL,
    text_detection_ms REAL,
    region_preprocess_ms REAL,
    ocr_ms REAL,
    tts_ms REAL,
    total_ms REAL,
    audio_path TEXT,
    error_code TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_reading_sessions_created_at
    ON reading_sessions(created_at DESC);
"""

_HISTORY_COLUMNS = (
    "id",
    "created_at",
    "status",
    "input_kind",
    "original_image_path",
    "preprocessed_image_path",
    "recognized_text",
    "recognition_engine",
    "confidence_mean",
    "preprocess_ms",
    "text_detection_ms",
    "region_preprocess_ms",
    "ocr_ms",
    "tts_ms",
    "total_ms",
    "audio_path",
    "error_code",
    "error_message",
)


def _connect(database_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(database_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def initialize_database(database_path: str | Path = DATABASE_PATH) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as connection, connection:
        connection.executescript(_SCHEMA)
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(reading_sessions)")
        }
        for column in ("text_detection_ms", "region_preprocess_ms"):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE reading_sessions ADD COLUMN {column} REAL")
        if "recognition_engine" not in existing_columns:
            connection.execute(
                "ALTER TABLE reading_sessions ADD COLUMN recognition_engine TEXT"
            )


def record_reading(
    result: dict[str, Any],
    *,
    input_kind: str | None = None,
    database_path: str | Path = DATABASE_PATH,
) -> None:
    values = {
        "id": result.get("id") or uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": result.get("status", "failed"),
        "input_kind": input_kind or result.get("input_kind", "upload"),
        "original_image_path": _path_string(result.get("original_image_path")),
        "preprocessed_image_path": _path_string(result.get("preprocessed_image_path")),
        "recognized_text": result.get("recognized_text"),
        "recognition_engine": result.get("recognition_engine", "Tesseract"),
        "confidence_mean": result.get("confidence_mean"),
        "preprocess_ms": result.get("preprocess_ms"),
        "text_detection_ms": result.get("text_detection_ms"),
        "region_preprocess_ms": result.get("region_preprocess_ms"),
        "ocr_ms": result.get("ocr_ms"),
        "tts_ms": result.get("tts_ms"),
        "total_ms": result.get("total_ms"),
        "audio_path": _path_string(result.get("audio_path")),
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
    }
    placeholders = ", ".join(f":{column}" for column in _HISTORY_COLUMNS)
    columns = ", ".join(_HISTORY_COLUMNS)
    with closing(_connect(database_path)) as connection, connection:
        connection.execute(
            f"INSERT INTO reading_sessions ({columns}) VALUES ({placeholders})",
            values,
        )


def record_failure(
    message: str,
    *,
    input_kind: str,
    database_path: str | Path = DATABASE_PATH,
) -> None:
    record_reading(
        {
            "id": uuid.uuid4().hex,
            "status": "failed",
            "input_kind": input_kind,
            "error_code": "processing_error",
            "error_message": message,
        },
        database_path=database_path,
    )


def get_recent_readings(
    limit: int = 50,
    *,
    database_path: str | Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("History limit must be at least one.")
    if not Path(database_path).is_file():
        return []

    with closing(_connect(database_path)) as connection:
        rows = connection.execute(
            "SELECT * FROM reading_sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_reading_count(database_path: str | Path = DATABASE_PATH) -> int:
    if not Path(database_path).is_file():
        return 0
    with closing(_connect(database_path)) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM reading_sessions").fetchone()
    return int(row["count"])


def _path_string(value: str | Path | None) -> str | None:
    return str(value) if value is not None else None
