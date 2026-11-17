"""Pipeline: dedupe -> language check -> embed -> cluster per polarity -> label -> verify -> compare."""
from __future__ import annotations

import math
import random
import time
from collections import Counter

from .quotes import verify_all
from .text import dedupe, detect_language, tokens, unit

MIN_WORDS = 5          # shorter reviews ("good game") carry no theme
SAMPLE_PER_CLUSTER = 12
LOW_N = 8              # themes with fewer reviews are flagged, not ranked


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def kmeans(vecs: list[list[float]], k: int, seed: int = 7, iters: int = 30) -> list[int]:
    """Spherical k-means (cosine) with k-means++ seeding. Deterministic for a given seed."""
    n = len(vecs)
    k = max(1, min(k, n))
    rng = random.Random(seed)
    cents = [vecs[rng.randrange(n)]]
    while len(cents) < k:
        d = [1 - max(_dot(v, c) for c in cents) for v in vecs]
        total = sum(d)
        if total <= 0:
            break
        r, acc = rng.random() * total, 0.0
        for i, di in enumerate(d):
            acc += di
            if acc >= r:
                cents.append(vecs[i])
                break
    assign = [0] * n
    for _ in range(iters):
        new = [max(range(len(cents)), key=lambda j: _dot(v, cents[j])) for v in vecs]
        if new == assign and _ > 0:
            break
        assign = new
        for j in range(len(cents)):
            members = [vecs[i] for i in range(n) if assign[i] == j]
            if members:
                cents[j] = unit([sum(col) for col in zip(*members)])
    return assign


def auto_k(n: int) -> int:
    return max(2, min(7, round(math.sqrt(n / 4)))) if n >= 8 else 1


def two_prop_z(a: int, na: int, b: int, nb: int) -> float:
    """z statistic for the change in share from a/na to b/nb (pooled)."""
    if not na or not nb:
        return 0.0
    p = (a + b) / (na + nb)
    se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    return 0.0 if se == 0 else (b / nb - a / na) / se


def prepare(data: dict) -> tuple[list[dict], dict]:
    """Dedupe, language-check and split into before/after. Returns (usable reviews, stats)."""
    raw = data["reviews"]
    reviews, dup = dedupe(raw)
    patch_ts = data["patch"]["date"]
    langs, mismatches, usable, short = Counter(), [], [], 0
    for r in reviews:
        lang = detect_language(r["review"])
        langs[lang] += 1
        if lang not in ("english", "unknown") and r.get("language") == "english":
            mismatches.append({"id": r["id"], "steam_tag": r["language"], "detected": lang})
        if lang != "english":
            continue
        if len(tokens(r["review"])) < MIN_WORDS:
            short += 1
            continue
        usable.append({**r, "side": "before" if r["created"] < patch_ts else "after"})
    stats = {"raw": len(raw), "duplicates_removed": dup, "detected_languages": dict(langs.most_common()),
             "tag_mismatches": len(mismatches), "tag_mismatch_examples": mismatches[:10],
             "too_short": short, "analyzed": len(usable)}
    return usable, stats


def volume(data: dict) -> dict:
    """True review volume per window from Steam's per-day totals (not from the sample)."""
    out = {}
    for side in ("before", "after"):
        days = [d for d in data.get("days", []) if d["side"] == side]
        pos = sum(d.get("total_positive") or 0 for d in days)
        neg = sum(d.get("total_negative") or 0 for d in days)
        out[side] = {"days": len(days), "positive": pos, "negative": neg,
                     "negative_share": round(neg / (pos + neg), 4) if pos + neg else None,
                     "per_day": round((pos + neg) / len(days), 1) if days else None}
    return out


def analyze(data: dict, backend, k: int | None = None, seed: int = 7) -> dict:
    t0 = time.time()
    usable, stats = prepare(data)
    themes, all_quotes = [], []
    for polarity, voted_up in (("negative", False), ("positive", True)):
        group = [r for r in usable if r["voted_up"] == voted_up]
        if not group:
            continue
        vecs = [unit(v) for v in backend.embed([r["review"] for r in group])]
        assign = kmeans(vecs, k or auto_k(len(group)), seed=seed)
        side_n = Counter(r["side"] for r in group)
        for j in sorted(set(assign)):
            idx = [i for i, a in enumerate(assign) if a == j]
            cent = unit([sum(col) for col in zip(*(vecs[i] for i in idx))])
            idx.sort(key=lambda i: -_dot(vecs[i], cent))
            members = [group[i] for i in idx]
            # sample both sides so the label is not written from one window only
            before = [m for m in members if m["side"] == "before"]
            after = [m for m in members if m["side"] == "after"]
            sample = [m for pair in zip(before, after) for m in pair][:SAMPLE_PER_CLUSTER]
            sample += [m for m in members if m not in sample][:SAMPLE_PER_CLUSTER - len(sample)]
            label = backend.label({"game": data["name"], "polarity": polarity, "sample": sample,
                                   "background": [r["review"] for r in group if r not in members]})
            texts = {m["id"]: m["review"] for m in members}
            side_of = {m["id"]: m["side"] for m in members}
            for q in verify_all(label["quotes"], texts) + verify_all(label["feature_requests"], texts):
                q["side"] = side_of.get(q["review_id"])
            nb, na = len(before), len(after)
            sb = nb / side_n["before"] if side_n["before"] else 0.0
            sa = na / side_n["after"] if side_n["after"] else 0.0
            z = two_prop_z(nb, side_n["before"], na, side_n["after"])
            trend = "low n" if nb + na < LOW_N else ("rose" if z >= 1.96 else "fell" if z <= -1.96 else "flat")
            theme = {
                "id": f"{polarity[:3]}-{j}", "polarity": polarity, **{k_: label[k_] for k_ in
                                                                      ("category", "theme", "summary")},
                "label_parsed": label["parse_ok"], "labelled_by": label.get("model", backend.name),
                "size": len(members),
                "before": {"count": nb, "share": round(sb, 4)}, "after": {"count": na, "share": round(sa, 4)},
                "delta_pp": round((sa - sb) * 100, 1), "z": round(z, 2), "trend": trend,
                "quotes": label["quotes"], "feature_requests": label["feature_requests"],
                "review_ids": [m["id"] for m in members],
            }
            themes.append(theme)
            all_quotes += [dict(q, theme=theme["id"]) for q in label["quotes"] + label["feature_requests"]]
    unverified = [q for q in all_quotes if not q["verified"]]
    return {
        "game": data["name"], "appid": data["appid"], "patch": data["patch"],
        "window_days": data.get("window_days"), "window_note": data.get("note", ""),
        "source": data.get("source"), "sampling": data.get("sampling"),
        "collected_at": data.get("collected_at"), "backend": backend.name,
        "volume": volume(data), "stats": stats,
        "sample_sizes": {p: dict(Counter(r["side"] for r in usable if r["voted_up"] == (p == "positive")))
                         for p in ("negative", "positive")},
        "themes": sorted(themes, key=lambda t: (t["polarity"], -t["size"])),
        "verification": {"total": len(all_quotes), "verified": len(all_quotes) - len(unverified),
                         "unverified": len(unverified), "unverified_items": unverified},
        "elapsed_s": round(time.time() - t0, 1),
    }
