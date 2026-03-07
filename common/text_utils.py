"""Text preprocessing utilities shared across v0/v1/v2.

The helpers in this module implement lightweight normalization for raw review text.
They are used consistently across training and inference to improve reproducibility.
"""

import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
IMDB_NEWLINE_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
SPECIAL_RE = re.compile(r"[^0-9a-zA-Z\s]+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize raw review text for vectorization.

    This function performs light cleaning (e.g., lowercasing and removing/normalizing
    punctuation/whitespace) while preserving the meaning of the review.

    Args:
        text: Raw input string.

    Returns:
        A cleaned string suitable for downstream tokenization/vectorization.
    """
    text = str(text).lower()
    text = IMDB_NEWLINE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text
