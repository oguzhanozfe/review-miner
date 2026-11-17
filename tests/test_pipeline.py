import json
import unittest
from pathlib import Path
from unittest import mock

from review_miner import steam
from review_miner.analyze import analyze, kmeans, two_prop_z, volume
from review_miner.llm import OfflineBackend, parse_label
from review_miner.quotes import verify_all, verify_quote
from review_miner.report import render_html
from review_miner.text import dedupe, detect_language, normalize, tfidf_vectors

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = sorted((ROOT / "fixtures").glob("*.json"))


def rev(i, text, up=False, created=1000, lang="english"):
    return {"id": str(i), "review": text, "voted_up": up, "created": created, "language": lang,
            "votes_up": 0, "playtime_at_review_min": 0}


class TextTests(unittest.TestCase):
    def test_normalize_handles_crlf_curly_quotes_bbcode(self):
        self.assertEqual(normalize("It’s  [b]BROKEN[/b]\r\nnow"), "it's broken now")

    def test_dedupe_ids_and_copypasta(self):
        rs = [rev(1, "Great game"), rev(1, "other"), rev(2, "great  GAME"), rev(3, "fine")]
        out, removed = dedupe(rs)
        self.assertEqual([r["id"] for r in out], ["1", "3"])
        self.assertEqual(removed, 2)

    def test_detect_language(self):
        self.assertEqual(detect_language("The game is great and it runs well on my deck"), "english")
        self.assertEqual(detect_language("Отличная игра, очень советую всем"), "russian")
        self.assertEqual(detect_language("o jogo é muito bom e não tem bugs"), "brazilian")
        self.assertEqual(detect_language("gg"), "unknown")

    def test_tfidf_vectors_unit_and_deterministic(self):
        a = tfidf_vectors(["crash on startup every time", "lovely farming"])
        b = tfidf_vectors(["crash on startup every time", "lovely farming"])
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a[0]), 1.0, places=6)


class QuoteTests(unittest.TestCase):
    TEXT = "Update 2.0 is great.\r\nBut the game crashes every time I open the map, please fix."

    def test_verbatim_quote_passes_after_normalisation(self):
        ok, _ = verify_quote("“…the game CRASHES every time I open the map…”", self.TEXT)
        self.assertTrue(ok)

    def test_fabricated_quote_is_reported(self):
        items = [{"review_id": "1", "quote": "the game crashes every time I load a save"},
                 {"review_id": "1", "quote": "crashes every time I open the map"},
                 {"review_id": "9", "quote": "crashes every time I open the map"}]
        verify_all(items, {"1": self.TEXT})
        self.assertEqual([i["verified"] for i in items], [False, True, False])
        self.assertIn("not found", items[0]["reason"])
        self.assertIn("not in this cluster", items[2]["reason"])

    def test_stitched_and_tiny_quotes_fail(self):
        self.assertFalse(verify_quote("Update 2.0 ... please fix", self.TEXT)[0])
        self.assertFalse(verify_quote("great", self.TEXT)[0])


class ParseTests(unittest.TestCase):
    def test_parse_label_tolerates_fences_and_bad_category(self):
        raw = '```json\n{"category": "Lag", "theme": "x", "quotes": [{"review_id": 5, "quote": "q"}]}\n```'
        lab = parse_label(raw)
        self.assertEqual(lab["category"], "other")
        self.assertEqual(lab["quotes"][0]["review_id"], "5")
        self.assertFalse(parse_label("not json")["parse_ok"])


class StatsTests(unittest.TestCase):
    def test_kmeans_separates_obvious_groups(self):
        vecs = [[1, 0], [0.99, 0.14], [0, 1], [0.14, 0.99]]
        a = kmeans(vecs, 2, seed=1)
        self.assertEqual(a[0], a[1])
        self.assertEqual(a[2], a[3])
        self.assertNotEqual(a[0], a[2])

    def test_two_prop_z(self):
        self.assertGreater(two_prop_z(5, 100, 30, 100), 1.96)
        self.assertEqual(two_prop_z(0, 0, 3, 10), 0.0)

    def test_volume_uses_steam_totals(self):
        v = volume({"days": [{"side": "before", "total_positive": 90, "total_negative": 10},
                             {"side": "after", "total_positive": 50, "total_negative": 50}]})
        self.assertEqual(v["before"]["negative_share"], 0.1)
        self.assertEqual(v["after"]["per_day"], 100.0)


class FakeBackend(OfflineBackend):
    """Offline embeddings, but labels that include one fabricated quote per cluster."""
    name = "fake"

    def label(self, cluster):
        lab = super().label(cluster)
        lab["quotes"].append({"review_id": cluster["sample"][0]["id"], "quote": "this sentence was never written by anyone"})
        return lab


class PipelineTests(unittest.TestCase):
    def synthetic(self):
        neg = ["The game crashes on startup every single time since the patch",
               "Crashes constantly after the update, crashes on the loading screen",
               "Terrible performance, fps drops to 20 and the stutter is awful",
               "Stutter and fps drops everywhere, the performance is unplayable now"]
        pos = ["Wonderful farming and lovely relaxing music, I adore this game",
               "Relaxing farming game with lovely music and great characters"]
        rs = [rev(i, t, created=500 if i % 2 else 1500) for i, t in enumerate(neg * 3)]
        rs += [rev(100 + i, t + f" {'!' * i}", up=True, created=500 if i % 2 else 1500) for i, t in enumerate(pos * 3)]
        rs.append(rev(999, "o jogo é muito bom e não tem bugs nenhum", up=True))
        return {"name": "Test Game", "appid": 1, "patch": {"date": 1000, "title": "Patch", "url": "http://x", "gid": "1"},
                "window_days": {"before": 1, "after": 1}, "days": [], "reviews": rs}

    def test_fabricated_quote_lands_in_unverified(self):
        res = analyze(self.synthetic(), FakeBackend(), k=2)
        unv = res["verification"]["unverified_items"]
        self.assertTrue(unv)
        self.assertTrue(all(q["quote"] == "this sentence was never written by anyone" for q in unv))
        self.assertEqual(res["stats"]["tag_mismatches"], 1)
        html = render_html(res)
        self.assertIn("this sentence was never written by anyone", html)
        self.assertIn("UNVERIFIED", html)

    def test_counts_add_up(self):
        res = analyze(self.synthetic(), OfflineBackend(), k=2)
        for pol in ("negative", "positive"):
            ts = [t for t in res["themes"] if t["polarity"] == pol]
            self.assertEqual(sum(t["before"]["count"] for t in ts), res["sample_sizes"][pol]["before"])
            self.assertAlmostEqual(sum(t["after"]["share"] for t in ts), 1.0, places=3)
        self.assertEqual(res["verification"]["unverified"], 0)
        for c in res["categories"]:  # category rollup is the sum of its clusters
            ts = [t for t in res["themes"] if (t["polarity"], t["category"]) == (c["polarity"], c["category"])]
            self.assertEqual(c["after"]["count"], sum(t["after"]["count"] for t in ts))


@unittest.skipUnless(FIXTURES, "no recorded fixtures")
class FixtureTests(unittest.TestCase):
    def test_recorded_fixtures_run_offline_and_carry_no_author_ids(self):
        for path in FIXTURES:
            data = json.loads(path.read_text())
            self.assertGreaterEqual(len(data["reviews"]), 100, path.name)
            self.assertNotIn("steamid", path.read_text(), path.name)
            res = analyze(data, OfflineBackend())
            self.assertEqual(res["verification"]["unverified"], 0, path.name)
            self.assertTrue(any(t["polarity"] == "negative" for t in res["themes"]))


class SteamTests(unittest.TestCase):
    def test_slim_review_drops_author_identity(self):
        raw = {"recommendationid": "42", "language": "english", "review": "ok", "timestamp_created": 5,
               "voted_up": True, "author": {"steamid": "7656", "playtime_at_review": 90},
               "hardware": {"gpu": "x"}}
        slim = steam.slim_review(raw)
        self.assertNotIn("7656", json.dumps(slim))
        self.assertEqual(slim["playtime_at_review_min"], 90)

    def test_cursor_pagination_and_totals(self):
        pages = [{"response": {"reviews": [{"recommendationid": "1", "timestamp_created": 1}], "cursor": "A",
                               "total_matching": 7}},
                 {"response": {"reviews": [{"recommendationid": "2", "timestamp_created": 2}], "cursor": "B"}},
                 {"response": {"reviews": [], "cursor": "C"}}]
        calls = []

        def fake(url, limiter):
            calls.append(url)
            return pages[len(calls) - 1]
        with mock.patch.object(steam, "get_json", fake):
            out, total = steam.fetch_reviews(1, 0, 10, steam.RateLimiter(0), pages=5)
        self.assertEqual([r["id"] for r in out], ["1", "2"])
        self.assertEqual(total, 7)
        self.assertIn("%22cursor%22%3A%20%22A%22", calls[1])
        self.assertEqual(len(calls), 3)

    def test_redirect_is_an_error_not_followed(self):
        err = steam.urllib.error.HTTPError("u", 302, "Found", {}, None)
        with mock.patch.object(steam._opener, "open", side_effect=err):
            with self.assertRaisesRegex(RuntimeError, "redirected"):
                steam.get_json("http://x", steam.RateLimiter(0))


class CompatBackendTests(unittest.TestCase):
    def test_429_falls_back_to_next_model_and_caches(self):
        import tempfile
        from review_miner import llm
        calls = []

        def fake_post(url, payload, headers=None, **kw):
            calls.append(payload["model"])
            if payload["model"] == "first":
                raise llm.QuotaError("429")
            return {"choices": [{"message": {"content": '<thought>hmm {"x":1}</thought>{"category": "bug", "theme": "t"}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test"}), \
                mock.patch.object(llm, "_post", fake_post):
            b = llm.OpenAICompatBackend("gemini", cache=llm.DiskCache(d))
            b.models = ["first", "second"]
            cluster = {"game": "G", "polarity": "negative", "sample": [rev(1, "it crashes a lot")]}
            lab = b.label(cluster)
            self.assertEqual((lab["category"], lab["model"]), ("bug", "second"))
            again = b.label(cluster)  # served from the disk cache
            self.assertEqual(again["theme"], "t")
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(b.usage["cache_hits"], 1)


class DemoTests(unittest.TestCase):
    def test_no_key_falls_back_to_replay_and_never_calls_network(self):
        from review_miner import llm
        env = {k: "" for k in ("GEMINI_API_KEY", "OPENAI_API_KEY")}
        with mock.patch.dict("os.environ", env), mock.patch.object(llm, "ollama_ready", lambda: False), \
                mock.patch.object(llm, "_post", side_effect=AssertionError("network")):
            b, line = llm.resolve_demo_backend()
            self.assertTrue(line.startswith("mode: replay"))
            data = json.loads((ROOT / "fixtures" / "stardew-valley-1.6.json").read_text())
            res = analyze(data, b)
        ex = json.loads((ROOT / "examples" / "stardew-valley-1.6" / "report.json").read_text())
        self.assertEqual(res["themes"], ex["themes"])

    def test_replay_miss_is_a_clear_error(self):
        from review_miner import llm
        b = llm.OpenAICompatBackend("gemini", cache=llm.DiskCache(llm.REPLAY_DIR), replay=True)
        with self.assertRaises(SystemExit):
            b.embed(["a review that was never recorded"])


if __name__ == "__main__":
    unittest.main()
