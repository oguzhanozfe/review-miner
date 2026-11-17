"""Quote verification: every quote must appear verbatim in the review it cites."""
from __future__ import annotations

import re

from .text import normalize

_EDGE = re.compile(r'^[\s"\'.…]+|[\s"\'.…]+$')
MIN_QUOTE_CHARS = 12


def clean_quote(q: str) -> str:
    """Strip wrapping quote marks and leading/trailing ellipses that models add."""
    return _EDGE.sub("", normalize(q))


def verify_quote(quote: str, review_text: str | None) -> tuple[bool, str]:
    """True only if the normalised quote is a substring of the normalised cited review.

    Both sides go through text.normalize (NFKC, ASCII quotes, BBCode stripped, whitespace
    collapsed, casefolded); the quote additionally loses wrapping quotes and edge ellipses.
    Inner ellipses are NOT bridged: a quote stitched from two places fails.
    """
    if review_text is None:
        return False, "cited review id not in this cluster"
    q = clean_quote(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return False, f"quote shorter than {MIN_QUOTE_CHARS} characters"
    if q in normalize(review_text):
        return True, "verbatim"
    return False, "text not found in cited review"


def verify_all(items: list[dict], texts_by_id: dict[str, str]) -> list[dict]:
    """Annotate each {'quote', 'review_id'} item with verified/reason; returns the same list."""
    for it in items:
        ok, why = verify_quote(it.get("quote", ""), texts_by_id.get(str(it.get("review_id", ""))))
        it["verified"], it["reason"] = ok, why
    return items
