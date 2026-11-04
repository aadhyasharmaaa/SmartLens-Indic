from __future__ import annotations

import unicodedata


def normalize_unicode_text(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)
