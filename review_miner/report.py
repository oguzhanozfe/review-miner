"""Static single-file HTML report. No JavaScript, no external assets."""
from __future__ import annotations

import datetime as dt
from html import escape

CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--up:#b3261e;--down:#1b6e3a;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a09f99;--line:#33322f;--card:#1f1f1d;--up:#ff8a80;--down:#7fd69a;--warn:#f2c46d}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:.2em 0}h2{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.2em}
.mut{color:var(--mut)}.note{border-left:3px solid var(--warn);padding:.4em .8em;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.theme{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:12px 0}
.tag{display:inline-block;font-size:.75rem;border:1px solid var(--line);border-radius:4px;padding:0 6px;margin-right:4px}
.rose{color:var(--up);font-weight:600}.fell{color:var(--down);font-weight:600}
blockquote{margin:.4em 0;padding:.2em .8em;border-left:3px solid var(--line)}
.bad{border-left-color:var(--up)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.kpi b{font-size:1.3rem;display:block}
"""


def _date(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d") if ts else "?"


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _quote(q, show_request=False):
    mark = "verified" if q["verified"] else f"UNVERIFIED: {escape(q['reason'])}"
    side = f"<span class=tag>{q['side']}</span>" if q.get("side") else ""
    head = f"<b>{escape(q['request'])}</b><br>" if show_request else ""
    return (f"<blockquote class='{'' if q['verified'] else 'bad'}'>{head}“{escape(q['quote'])}”"
            f"<br><span class=mut>{side}review {escape(q['review_id'])} &middot; {mark}</span></blockquote>")


def _theme(t):
    trend = t["trend"]
    cls = trend if trend in ("rose", "fell") else ""
    fr = "".join(_quote(f, True) for f in t["feature_requests"])
    return f"""<div class=theme id="{t['id']}">
<span class=tag>{escape(t['category'])}</span><span class=tag>{t['polarity']}</span>
<b>{escape(t['theme'])}</b> <span class=mut>({t['size']} reviews &middot; labelled by {escape(str(t.get('labelled_by', '')))})</span>
<p>{escape(t['summary'])}</p>
<p>Before: {t['before']['count']} ({_pct(t['before']['share'])} of sampled {t['polarity']} reviews) &rarr;
After: {t['after']['count']} ({_pct(t['after']['share'])}) &middot;
<span class="{cls}">{t['delta_pp']:+.1f} pp, {trend}</span> <span class=mut>(z={t['z']})</span></p>
{''.join(_quote(q) for q in t['quotes'])}
{('<p class=mut>Feature requests</p>' + fr) if fr else ''}
</div>"""


def _table(themes):
    rows = "".join(
        f"<tr><td><a href='#{t['id']}'>{escape(t['theme'])}</a></td><td>{escape(t['category'])}</td>"
        f"<td>{t['before']['count']} / {_pct(t['before']['share'])}</td>"
        f"<td>{t['after']['count']} / {_pct(t['after']['share'])}</td>"
        f"<td class='{t['trend'] if t['trend'] in ('rose', 'fell') else ''}'>{t['delta_pp']:+.1f} pp {t['trend']}</td></tr>"
        for t in sorted(themes, key=lambda t: -abs(t["delta_pp"])))
    return ("<table><tr><th>Theme</th><th>Category</th><th>Before (n / share)</th>"
            f"<th>After (n / share)</th><th>Change</th></tr>{rows}</table>")


def _cat_table(cats, polarity):
    rows = "".join(
        f"<tr><td>{escape(c['category'])}</td><td>{c['themes']}</td>"
        f"<td>{c['before']['count']} / {_pct(c['before']['share'])}</td>"
        f"<td>{c['after']['count']} / {_pct(c['after']['share'])}</td>"
        f"<td class='{c['trend'] if c['trend'] in ('rose', 'fell') else ''}'>{c['delta_pp']:+.1f} pp {c['trend']}"
        f" <span class=mut>(z={c['z']})</span></td></tr>"
        for c in cats if c["polarity"] == polarity)
    return ("<table><tr><th>Category</th><th>Clusters</th><th>Before (n / share)</th>"
            f"<th>After (n / share)</th><th>Change</th></tr>{rows}</table>")


def render_html(r: dict) -> str:
    v, s, vol, ss = r["verification"], r["stats"], r["volume"], r["sample_sizes"]
    neg = [t for t in r["themes"] if t["polarity"] == "negative"]
    pos = [t for t in r["themes"] if t["polarity"] == "positive"]
    unv = "".join(_quote(q, "request" in q) + f"<p class=mut>theme {q['theme']}</p>"
                  for q in v["unverified_items"]) or "<p>None.</p>"
    p = r["patch"]
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{escape(r['game'])} review themes</title><style>{CSS}</style></head><body><main>
<p class=mut>review-miner report</p>
<h1>{escape(r['game'])}: reviews before vs after &ldquo;{escape(p['title'])}&rdquo;</h1>
<p>Patch posted {_date(p['date'])} (<a href="{escape(p['url'])}">Steam news</a>). Windows: {r['window_days']['before']} days before,
{r['window_days']['after']} days after. {escape(r.get('window_note') or '')}</p>
<p class=note>Labels by <b>{escape(r['backend'])}</b>. Quotes: <b>{v['verified']}/{v['total']}</b> verified verbatim against the cited review;
{v['unverified']} unverified (listed at the end, never silently dropped).
Reviews from <code>{escape(r['source'] or '')}</code>; Valve retired the old <code>/appreviews?json=1</code> endpoint in October 2026.</p>

<h2>Volume (all English reviews, from Steam's own counts)</h2>
<div class=grid>
<div class=kpi><span class=mut>Before: reviews / day</span><b>{vol['before']['per_day']}</b></div>
<div class=kpi><span class=mut>After: reviews / day</span><b>{vol['after']['per_day']}</b></div>
<div class=kpi><span class=mut>Before: negative share</span><b>{_pct(vol['before']['negative_share'])}</b></div>
<div class=kpi><span class=mut>After: negative share</span><b>{_pct(vol['after']['negative_share'])}</b></div>
</div>

<h2>Sample</h2>
<p>{s['raw']} reviews sampled (same budget every day, positives and negatives sampled separately, so
theme shares are <i>within</i> a polarity, not of all reviews). {s['duplicates_removed']} duplicates removed,
{s['tag_mismatches']} tagged English but detected as another language, {s['too_short']} too short to theme,
<b>{s['analyzed']}</b> analysed: negative {ss['negative'].get('before', 0)} before / {ss['negative'].get('after', 0)} after,
positive {ss['positive'].get('before', 0)} before / {ss['positive'].get('after', 0)} after.</p>
<p class=mut>Trend: &ldquo;rose&rdquo;/&ldquo;fell&rdquo; only when a two-proportion z-test passes |z| &ge; 1.96; themes under 8 reviews are &ldquo;low n&rdquo;.</p>

<h2>Complaints by category</h2>
<p class=mut>Clusters summed by label. k-means can split one topic into several clusters, so read this level first.</p>
{_cat_table(r.get('categories', []), 'negative')}
<h2>Praise by category</h2>{_cat_table(r.get('categories', []), 'positive')}
<h2>Complaint clusters: what changed</h2>{_table(neg)}
<h2>Praise clusters: what changed</h2>{_table(pos)}
<h2>Complaint themes</h2>{''.join(_theme(t) for t in neg)}
<h2>Praise themes</h2>{''.join(_theme(t) for t in pos)}
<h2>Unverified quotes</h2>{unv}
<p class=mut>Generated in {r['elapsed_s']} s. Data collected {_date(r.get('collected_at'))}.</p>
</main></body></html>"""
