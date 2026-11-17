"""Command line: `collect` records Steam fixtures, `analyze` turns one into a report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_collect(args) -> None:
    from .steam import RateLimiter, collect_game

    games = json.loads(Path(args.games).read_text())
    if args.only:
        games = [g for g in games if g["slug"] in args.only]
    limiter = RateLimiter(args.interval)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for g in games:
        data = collect_game(g, limiter, per_page=args.per_page, pos_pages=args.pos_pages,
                            neg_pages=args.neg_pages)
        path = Path(args.out) / f"{g['slug']}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(f"{g['slug']}: {len(data['reviews'])} reviews, patch '{data['patch']['title']}' -> {path}")


def cmd_analyze(args) -> None:
    from .analyze import analyze
    from .llm import make_backend
    from .report import render_html

    data = json.loads(Path(args.fixture).read_text())
    backend = make_backend(args.backend, model=args.model, embed_model=args.embed_model)
    result = analyze(data, backend, k=args.k, seed=args.seed)
    if getattr(backend, "usage", None):
        result["token_usage"] = backend.usage
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (out / "report.html").write_text(render_html(result))
    v = result["verification"]
    print(f"{out}/report.html  themes={len(result['themes'])}  quotes verified "
          f"{v['verified']}/{v['total']}  unverified={v['unverified']}")
    if args.strict and v["unverified"]:
        sys.exit(2)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="review-miner")
    sub = p.add_subparsers(required=True)

    c = sub.add_parser("collect", help="record a review fixture from Steam (network)")
    c.add_argument("--games", default="games.json")
    c.add_argument("--out", default="fixtures")
    c.add_argument("--only", nargs="*")
    c.add_argument("--per-page", type=int, default=10)
    c.add_argument("--pos-pages", type=int, default=1, help="positive cursor pages per day")
    c.add_argument("--neg-pages", type=int, default=2, help="negative cursor pages per day")
    c.add_argument("--interval", type=float, default=1.5, help="seconds between requests")
    c.set_defaults(func=cmd_collect)

    a = sub.add_parser("analyze", help="build report.html + report.json from a fixture")
    a.add_argument("fixture")
    a.add_argument("--out", default="out")
    a.add_argument("--backend", default="gemini", choices=["gemini", "deepseek", "openai", "ollama", "offline"],
                   help="gemini (default, GEMINI_API_KEY), deepseek (DEEPSEEK_API_KEY), openai "
                        "(OPENAI_API_KEY / OPENAI_BASE_URL), ollama (local), offline (no model)")
    a.add_argument("--model", default=None)
    a.add_argument("--embed-model", default=None)
    a.add_argument("--k", type=int, default=None, help="clusters per polarity (default: auto)")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--strict", action="store_true", help="exit 2 if any quote fails verification")
    a.set_defaults(func=cmd_analyze)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
