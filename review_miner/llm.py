"""Model backends. Each has embed(texts) -> vectors and label(cluster) -> dict.

- gemini  (default): Gemini through its OpenAI-compatible endpoint, key from GEMINI_API_KEY.
- deepseek: DeepSeek chat (DEEPSEEK_API_KEY); it has no embeddings API, so TF-IDF vectors.
- openai : any OpenAI-compatible API; OPENAI_API_KEY, base from OPENAI_BASE_URL.
- ollama : local Ollama, chat + embeddings, no key.
- offline: no model at all. TF-IDF vectors and a keyword lexicon; quotes are extracted
           sentences, so they verify by construction. Used by tests and the offline demo.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from .text import tfidf_vectors, normalize, top_terms

CATEGORIES = ["bug", "balance", "monetization", "performance", "content", "ux", "other"]

PROMPT = """You label one cluster of Steam reviews for {game} ({polarity} reviews).
Reviews, each prefixed by its id:
{reviews}

Return JSON only, with these keys:
"category": exactly one of {cats}. bug = crashes, glitches, broken things. balance = difficulty, weapon/enemy tuning, economy tuning. monetization = price, DLC, microtransactions, value. performance = FPS, stutter, loading, hardware. content = amount/quality of things to do, story, features. ux = UI, controls, menus, accounts, onboarding. other = none of these.
"theme": a short name for what these reviews share (max 6 words).
"summary": one sentence.
"quotes": 2 or 3 objects {{"review_id": "...", "quote": "..."}}. Each quote MUST be copied character for character from that review: one contiguous span of 5 to 25 words. Do not fix spelling, do not join two places, do not paraphrase.
"feature_requests": objects {{"request": "...", "review_id": "...", "quote": "..."}} for concrete things reviewers ask to be added or changed, with a verbatim supporting quote. Skip jokes, memes and sarcasm: Steam reviews are often ironic. Empty list if none."""


class QuotaError(RuntimeError):
    """HTTP 429: the model's quota is spent. Never retried in a loop; the caller switches model."""


def _post(url: str, payload: dict, headers: dict | None = None, timeout: int = 120,
          retries: int = 3) -> dict:
    """POST JSON. Retries timeouts and 5xx (busy servers); a 429 raises QuotaError at once."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (OSError, http.client.HTTPException) as e:  # URLError, resets, truncated bodies
            code = getattr(e, "code", None)
            if code == 429:
                raise QuotaError(f"429 from {url}") from e
            if attempt == retries or (code is not None and code < 500):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


class DiskCache:
    """Model responses on disk keyed by a hash of the exact request, so reruns cost nothing."""

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("REVIEW_MINER_CACHE", ".cache/review-miner"))

    def _path(self, key: dict) -> Path:
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.root / h[:2] / f"{h}.json"

    def get(self, key: dict):
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: dict, value) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))


_THOUGHT = re.compile(r"<thought>.*?</thought>", re.S)


def build_prompt(cluster: dict) -> str:
    lines = [f"[{r['id']}] {r['review'][:700].strip()}" for r in cluster["sample"]]
    return PROMPT.format(game=cluster["game"], polarity=cluster["polarity"],
                         reviews="\n\n".join(lines), cats=", ".join(CATEGORIES))


def parse_label(raw: str) -> dict:
    """Parse and shape-check a model reply; tolerate code fences and stray text around JSON."""
    m = re.search(r"\{.*\}", _THOUGHT.sub("", raw), re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    cat = str(data.get("category", "other")).strip().lower()
    return {
        "category": cat if cat in CATEGORIES else "other",
        "theme": str(data.get("theme", "")).strip()[:80] or "(unnamed)",
        "summary": str(data.get("summary", "")).strip()[:400],
        "quotes": [{"review_id": str(q.get("review_id", "")), "quote": str(q.get("quote", ""))}
                   for q in data.get("quotes", []) if isinstance(q, dict)][:3],
        "feature_requests": [{"request": str(f.get("request", ""))[:200],
                              "review_id": str(f.get("review_id", "")),
                              "quote": str(f.get("quote", ""))}
                             for f in data.get("feature_requests", []) if isinstance(f, dict)][:5],
        "parse_ok": bool(data),
    }


class OllamaBackend:
    def __init__(self, model=None, embed_model=None, host=None):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.model = model or os.environ.get("REVIEW_MINER_MODEL", "qwen3.5:9b")
        self.embed_model = embed_model or os.environ.get("REVIEW_MINER_EMBED_MODEL", "qwen3-embedding:0.6b")
        self.name = f"ollama:{self.model}@{self._digest(self.model)} + {self.embed_model}@{self._digest(self.embed_model)}"

    def _digest(self, model):
        """Short model digest from /api/tags, so a report names the exact weights it used."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as resp:
                tags = json.load(resp)["models"]
            return next((m["digest"][:12] for m in tags if m["name"] == model), "unknown")
        except (OSError, ValueError, KeyError):
            return "unknown"

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 32):
            batch = [t[:2000] for t in texts[i:i + 32]]
            out += _post(f"{self.host}/api/embed", {"model": self.embed_model, "input": batch})["embeddings"]
        return out

    def label(self, cluster):
        resp = _post(f"{self.host}/api/chat", {
            "model": self.model, "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0, "seed": 7},
            "messages": [{"role": "user", "content": build_prompt(cluster)}]})
        return parse_label(resp["message"]["content"])


PROVIDERS = {  # OpenAI-compatible chat endpoints; the key is read from the named env var only
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY",
               "models": ["gemini-3.5-flash-lite", "gemma-4-31b-it", "gemini-3.1-flash-lite"], "embed_model": "gemini-embedding-001",
               "embed_dims": 768},  # 768 of 3072 dims keeps pure-Python k-means fast
    "deepseek": {"base": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY",
                 "models": ["deepseek-chat"], "embed_model": None},  # no embeddings API: TF-IDF instead
    "openai": {"base": None, "key_env": "OPENAI_API_KEY", "models": ["gpt-4o-mini"],
               "embed_model": "text-embedding-3-small"},
}


class OpenAICompatBackend:
    """Any OpenAI-compatible provider.

    Free tiers have small daily quotas, so: every response is cached on disk by request hash,
    and a 429 moves on to the next model in the provider's list instead of retrying. Each label
    records which model wrote it; token usage is tracked so runs can be costed.
    """

    def __init__(self, provider="gemini", model=None, embed_model=None, cache=None):
        cfg = PROVIDERS[provider]
        self.key = os.environ.get(cfg["key_env"])
        if not self.key:
            raise SystemExit(f"{cfg['key_env']} is not set (or use --backend offline)")
        self.base = (os.environ.get("OPENAI_BASE_URL") if provider == "openai" else None) \
            or cfg["base"] or "https://api.openai.com/v1"
        self.base = self.base.rstrip("/")
        chosen = model or os.environ.get("REVIEW_MINER_MODEL")
        self.models = [chosen] if chosen else list(cfg["models"])
        self.embed_model = embed_model or os.environ.get("REVIEW_MINER_EMBED_MODEL") or cfg["embed_model"]
        self.embed_dims = cfg.get("embed_dims")
        self.cache = cache or DiskCache()
        self.provider = provider
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "cache_hits": 0,
                      "by_model": {}}

    @property
    def name(self):
        used = sorted(self.usage["by_model"]) or self.models
        return f"{self.provider}:{', '.join(used)} + {self.embed_model or 'local hashed TF-IDF'}"

    def _h(self):
        return {"Authorization": f"Bearer {self.key}"}

    def embed(self, texts):
        if not self.embed_model:
            return tfidf_vectors(texts)
        out = []
        for i in range(0, len(texts), 64):
            batch = [t[:2000] for t in texts[i:i + 64]]
            key = {"embed": self.embed_model, "dims": self.embed_dims, "input": batch}
            vecs = self.cache.get(key)
            if vecs is None:
                payload = {"model": self.embed_model, "input": batch}
                if self.embed_dims:
                    payload["dimensions"] = self.embed_dims
                for attempt in range(4):  # embedding 429s are per-minute: wait it out, bounded
                    try:
                        r = _post(f"{self.base}/embeddings", payload, self._h())
                        break
                    except QuotaError:
                        if attempt == 3:
                            raise
                        time.sleep(65)
                # stable sort keeps input order when "index" is absent (Gemini omits it)
                vecs = [[round(x, 5) for x in d["embedding"]]
                        for d in sorted(r["data"], key=lambda d: d.get("index", 0))]
                self.cache.put(key, vecs)
            out += vecs
        return out

    def label(self, cluster):
        prompt = build_prompt(cluster)
        while self.models:
            model = self.models[0]
            key = {"chat": model, "prompt": prompt}
            content = self.cache.get(key)
            if content is not None:
                self.usage["cache_hits"] += 1
            else:
                try:
                    r = _post(f"{self.base}/chat/completions", {
                        "model": model, "temperature": 0, "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}]}, self._h())
                except (QuotaError, OSError, http.client.HTTPException) as e:
                    # quota spent, or still 5xx/timeouts after retries: try the next model
                    self.usage.setdefault("model_failures", []).append(f"{model}: {e}"[:200])
                    self.models.pop(0)
                    continue
                content = r["choices"][0]["message"]["content"]
                self.cache.put(key, content)
                u = r.get("usage") or {}
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                self.usage["calls"] += 1
            self.usage["by_model"][model] = self.usage["by_model"].get(model, 0) + 1
            return {**parse_label(content), "model": model}
        raise QuotaError("every model in the list failed (429/5xx/timeout); rerun later, cached labels are kept")


# --- offline ---------------------------------------------------------------------------------
LEXICON = {
    "bug": "bug bugs bugged crash crashes crashed crashing glitch glitches broken softlock freeze "
           "freezes error disconnect disconnects corrupted save fix fixed",
    "balance": "difficulty difficult hard easy nerf nerfed nerfs buff buffs balance balanced overpowered op "
               "weapon weapons enemies enemy spawn spawns grind grindy tuning",
    "monetization": "price priced expensive cheap money dlc microtransactions microtransaction pay "
                    "refund worth value sale premium warbonds super credits cash",
    "performance": "fps performance lag stutter stuttering optimization optimized optimised gpu cpu ram "
                   "frame frames loading load runs smooth deck",
    "content": "content story quests quest missions mission features update new endgame hours "
               "characters world map farm items",
    "ux": "ui menu menus controls controller keybind interface inventory account psn tutorial "
          "settings hud",
}
_LEX = {k: set(v.split()) for k, v in LEXICON.items()}
_REQUEST = re.compile(r"\b(please add|please make|wish (?:there|it|they|you)|should add|would love|"
                      r"needs? (?:a|an|more)|hope they|add (?:a|an|more|the option))\b", re.I)
_SENT = re.compile(r"[^.!?\n]+[.!?]?")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if 5 <= len(s.split()) <= 30]


class OfflineBackend:
    name = "offline: hashed TF-IDF + keyword lexicon (no model)"

    def embed(self, texts):
        return tfidf_vectors(texts)

    def label(self, cluster):
        sample = cluster["sample"]
        words = Counter(w for r in sample for w in set(normalize(r["review"]).split()))
        scores = {c: sum(words[w] for w in lex) for c, lex in _LEX.items()}
        cat = max(scores, key=scores.get) if any(scores.values()) else "other"
        terms = top_terms([r["review"] for r in sample], cluster.get("background", []))
        quotes = []
        for r in sample:  # sample is ordered nearest-to-centroid first
            sents = _sentences(r["review"])
            hit = next((s for s in sents if set(normalize(s).split()) & _LEX.get(cat, set())), None)
            if hit or sents:
                quotes.append({"review_id": r["id"], "quote": hit or sents[0]})
            if len(quotes) == 3:
                break
        requests = []
        for r in sample:
            for s in _sentences(r["review"]):
                if _REQUEST.search(s):
                    requests.append({"request": s, "review_id": r["id"], "quote": s})
        return {"category": cat, "theme": " / ".join(terms) or "(unnamed)",
                "summary": f"Keyword match: {cat}; distinctive terms: {', '.join(terms) or 'none'}.",
                "quotes": quotes, "feature_requests": requests[:5], "parse_ok": True}


BACKENDS = ["gemini", "deepseek", "openai", "ollama", "offline"]


def make_backend(kind: str, model=None, embed_model=None):
    if kind == "offline":
        return OfflineBackend()
    if kind == "ollama":
        return OllamaBackend(model, embed_model)
    return OpenAICompatBackend(kind, model, embed_model)
