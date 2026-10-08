"""Tai Dam text as the blt voice spells it, shared by the checks."""

import re
import unicodedata


def normalise(s: str) -> str:
    """NFC, lower case, eBible's marks mapped to the voice's own, and the
    rest of the punctuation as spaces."""
    s = unicodedata.normalize("NFC", s.lower())
    s = s.replace("ꞌ", "'").replace("‑", "‐").replace("#", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w'‐ ]", " ", s)).strip()


def without_marks(s: str) -> str:
    """Apostrophes and hyphens as spaces, for scoring a transcript."""
    return re.sub(r"\s+", " ", re.sub(r"['‐]", " ", s)).strip()
