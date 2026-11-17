"""Steam collection: reviews via IUserReviewsService/GetAppReviews, patch dates via ISteamNews.

Valve retired store.steampowered.com/appreviews/<appid>?json=1 around 2026-10-22: it now
answers 302 to the store front page. The documented replacement is
https://partner.steamgames.com/doc/webapi/IUserReviewsService#GetAppReviews, which takes the
same filters as enums inside `input_json` and needs no key. Redirects are never followed, so a
retired endpoint fails loudly instead of parsing the store HTML.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
USER_AGENT = "review-miner/0.1 (+https://github.com/oguzhanozfe/review-miner)"
FILTER_RECENT = 1
PURCHASE_ALL = 1
REVIEW_TYPE = {"positive": 1, "negative": 2}
DAY = 86400


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - refuse every redirect
        return None


_opener = urllib.request.build_opener(_NoRedirect)


class RateLimiter:
    def __init__(self, min_interval: float = 1.5):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def get_json(url: str, limiter: RateLimiter, retries: int = 4) -> dict:
    """GET a JSON URL politely: rate-limited, no redirects, exponential backoff on 429/5xx."""
    for attempt in range(retries + 1):
        limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with _opener.open(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                raise RuntimeError(f"{url} redirected ({e.code}); endpoint retired or moved") from e
            if (e.code == 429 or e.code >= 500) and attempt < retries:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    raise RuntimeError("unreachable")


def slim_review(r: dict) -> dict:
    """Keep only what the pipeline uses. Author SteamIDs and hardware are dropped on purpose."""
    return {
        "id": str(r["recommendationid"]),
        "language": r.get("language", ""),
        "review": r.get("review", ""),
        "created": int(r["timestamp_created"]),
        "voted_up": bool(r.get("voted_up")),
        "votes_up": int(r.get("votes_up", 0)),
        "playtime_at_review_min": int(r.get("author", {}).get("playtime_at_review", 0) or 0),
    }


def fetch_reviews(appid: int, start: int, end: int, limiter: RateLimiter, *, review_type: int = 0,
                  per_page: int = 10, pages: int = 2, languages: list[str] | None = None
                  ) -> tuple[list[dict], int | None]:
    """Most recent reviews created in [start, end], following the cursor for `pages` pages.

    Returns (reviews, total_matching): the second value is Steam's count of all reviews that
    match the query, i.e. the real volume the sample was drawn from.
    """
    out, cursor, seen_cursors, total = [], "*", set(), None
    for _ in range(pages):
        params = {
            "appid": appid, "filter": FILTER_RECENT, "languages": languages or ["english"],
            "review_type": review_type,
            "num_per_page": per_page, "cursor": cursor, "purchase_type": PURCHASE_ALL,
            "date_range_start": start, "date_range_end": end,
        }
        url = REVIEWS_URL + "?input_json=" + urllib.parse.quote(json.dumps(params))
        resp = get_json(url, limiter).get("response", {})
        batch = resp.get("reviews", [])
        if total is None:
            total = resp.get("total_matching")
        out.extend(slim_review(r) for r in batch)
        cursor = resp.get("cursor")
        if not batch or not cursor or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
    return out, total


def fetch_news_item(appid: int, gid: str, around: int, limiter: RateLimiter) -> dict:
    """Find one news post by gid near a timestamp; its date is the patch date."""
    url = NEWS_URL + "?" + urllib.parse.urlencode(
        {"appid": appid, "count": 50, "maxlength": 300, "enddate": around + 7 * DAY,
         "feeds": "steam_community_announcements"})
    for n in get_json(url, limiter)["appnews"]["newsitems"]:
        if str(n["gid"]) == str(gid):
            return {"gid": str(n["gid"]), "title": n["title"], "date": int(n["date"]),
                    "url": n["url"], "tags": n.get("tags", [])}
    raise LookupError(f"news gid {gid} not found for app {appid}")


def collect_game(game: dict, limiter: RateLimiter, *, pos_pages: int = 1, neg_pages: int = 2,
                 per_page: int = 10) -> dict:
    """Sample English-tagged reviews day by day, stratified by polarity, around a patch.

    Every day gets the same budget (one page of the newest positive reviews, up to two pages
    of the newest negative ones), so a launch-day spike cannot swamp the window and complaints
    are not drowned by a 98%-positive game. Steam's real per-day totals are stored next to the
    sample, so the report can state true volumes and true negative share.
    """
    patch = fetch_news_item(game["appid"], game["patch_gid"], game["approx_date"], limiter)
    reviews, seen, days_out = [], set(), []
    for side, days in (("before", range(-game["days_before"], 0)), ("after", range(0, game["days_after"]))):
        for d in days:
            start = patch["date"] + d * DAY
            day = {"side": side, "start": start}
            for pol, pages in (("positive", pos_pages), ("negative", neg_pages)):
                batch, total = fetch_reviews(game["appid"], start, start + DAY - 1, limiter,
                                             review_type=REVIEW_TYPE[pol], per_page=per_page, pages=pages)
                day[f"total_{pol}"] = total
                for r in batch:
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        reviews.append(r)
            days_out.append(day)
    return {
        "appid": game["appid"], "name": game["name"], "patch": patch,
        "window_days": {"before": game["days_before"], "after": game["days_after"]},
        "note": game.get("note", ""),
        "source": "IUserReviewsService/GetAppReviews/v1 (filter=Recent, purchase_type=All, "
                  "languages=english, per day: newest positive + newest negative pages)",
        "sampling": {"per_page": per_page, "positive_pages_per_day": pos_pages,
                     "negative_pages_per_day": neg_pages},
        "days": days_out, "collected_at": int(time.time()), "reviews": reviews,
    }
