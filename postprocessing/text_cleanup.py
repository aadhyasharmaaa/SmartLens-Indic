from __future__ import annotations

from postprocessing.normalize_unicode import normalize_unicode_text


def clean_ocr_text(text: str | None) -> str:
    if not text:
        return ""

    normalized = normalize_unicode_text(text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [" ".join(line.split()) for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line)
