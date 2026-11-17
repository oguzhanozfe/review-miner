"""Text utilities: normalisation, dedupe, a small language detector, hashed TF-IDF vectors."""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‛": "'", "′": "'",
                            "“": '"', "”": '"', "‟": '"', "″": '"',
                            "–": "-", "—": "-", "…": "..."})
_WS = re.compile(r"\s+")
_BBCODE = re.compile(r"\[/?[a-z0-9*]+(?:=[^\]]*)?\]", re.I)


def normalize(text: str) -> str:
    """The single normalisation used for dedupe and quote verification.

    NFKC, curly quotes/dashes/ellipsis to ASCII, Steam BBCode tags removed, whitespace
    (including \\r\\n) collapsed to one space, casefolded.
    """
    t = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    t = _BBCODE.sub(" ", t)
    return _WS.sub(" ", t).strip().casefold()


def dedupe(reviews: list[dict]) -> tuple[list[dict], int]:
    """Drop repeated review ids and exact copy-paste texts (after normalisation)."""
    seen_ids, seen_text, out = set(), set(), []
    for r in reviews:
        key = normalize(r["review"])
        if r["id"] in seen_ids or (key and key in seen_text):
            continue
        seen_ids.add(r["id"])
        seen_text.add(key)
        out.append(r)
    return out, len(reviews) - len(out)


# --- language detection -------------------------------------------------------------------
# Scripts settle most non-Latin languages; Latin-script ones are scored by stopword hits.
_SCRIPTS = [  # (steam language code, regex for the script)
    ("russian", re.compile(r"[Ѐ-ӿ]")),
    ("koreana", re.compile(r"[가-힯]")),
    ("japanese", re.compile(r"[぀-ヿ]")),
    ("schinese", re.compile(r"[一-鿿]")),
    ("thai", re.compile(r"[฀-๿]")),
    ("arabic", re.compile(r"[؀-ۿ]")),
    ("greek", re.compile(r"[Ͱ-Ͽ]")),
]
_STOP = {
    "english": "the and is it to of this you a in that for but with game not are be my have was "
               "on just so if its it's can all like i do they at what get",
    "spanish": "el la de que y en es un una los por con para no lo muy pero juego las del se",
    "brazilian": "o a de que e é um uma do da não com para jogo muito mas os se eu em",
    "german": "der die das und ist nicht ein eine ich es zu mit spiel aber auch sehr man sich",
    "french": "le la les et est un une de des pas je pour que en jeu mais très avec ce il",
    "italian": "il la di che e è un una non per gioco molto ma con sono anche lo gli",
    "turkish": "ve bir bu çok için ama oyun da de değil ile gibi daha ne var",
    "polish": "i w nie na jest to się z że gra do ale jak bardzo tak",
}
_STOPSETS = {k: set(v.split()) for k, v in _STOP.items()}
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def detect_language(text: str) -> str:
    """Return a Steam language code, or 'unknown' when the text is too short to call."""
    t = normalize(text)
    letters = sum(c.isalpha() for c in t)
    if letters < 3:
        return "unknown"
    for code, rx in _SCRIPTS:
        if len(rx.findall(t)) / letters > 0.3:
            if code == "schinese" and _SCRIPTS[2][1].search(t):
                return "japanese"
            return code
    words = tokens(t)
    if len(words) < 3:
        return "unknown"
    scores = {lang: sum(w in s for w in words) for lang, s in _STOPSETS.items()}
    best = max(scores, key=scores.get)
    if scores[best] < max(1, 0.1 * len(words)):
        return "unknown"
    return best


# --- hashed TF-IDF (the offline embedding) ---------------------------------------------------
_EN_STOP = _STOPSETS["english"] | set(
    "i me we our your he she him her them their there here was were been being has had having "
    "does did doing an or as from by about into than then too very can will would should could "
    "no yes more most some any one also only really even much many still now out up down over "
    "again game games play played playing it's i'm don't im dont 10 time get got make".split())


def content_tokens(text: str) -> list[str]:
    return [w for w in tokens(text) if w not in _EN_STOP and len(w) > 2 and not w.isdigit()]


def tfidf_vectors(texts: list[str], dim: int = 1024) -> list[list[float]]:
    """Unit-length hashed TF-IDF vectors over unigrams+bigrams. Deterministic, stdlib only."""
    docs = []
    for t in texts:
        toks = content_tokens(t)
        docs.append(Counter(toks + [a + " " + b for a, b in zip(toks, toks[1:])]))
    df = Counter(term for d in docs for term in d)
    n = len(docs)
    vecs = []
    for d in docs:
        v = [0.0] * dim
        for term, tf in d.items():
            h = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big")
            v[h % dim] += (1 + math.log(tf)) * math.log((1 + n) / (1 + df[term]))
        vecs.append(unit(v))
    return vecs


def unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def top_terms(texts: list[str], background: list[str], n: int = 4) -> list[str]:
    """Terms over-represented in `texts` relative to `background` (log-odds, add-one)."""
    fg = Counter(w for t in texts for w in set(content_tokens(t)))
    bg = Counter(w for t in background for w in set(content_tokens(t)))
    nf, nb = max(len(texts), 1), max(len(background), 1)
    scored = {w: c * math.log(((c + 1) / nf) / ((bg[w] + 1) / nb)) for w, c in fg.items() if c >= 2}
    return [w for w, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]
