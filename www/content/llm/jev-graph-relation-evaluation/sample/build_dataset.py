#!/usr/bin/env python3
"""Build a bilingual Japanese/English CoDEx-S relation-choice test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODEX_ROOT = ROOT.parent / "codex"
TEST_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/test.txt"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
JAWIKI_API = "https://ja.wikipedia.org/w/api.php"
LAST_REQUEST_AT = 0.0
USER_AGENT = "CodexBilingualRelationChoice/1.0"


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def get_json(url, params):
    global LAST_REQUEST_AT
    request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            delay = 1.1 - (time.monotonic() - LAST_REQUEST_AT)
            if delay > 0:
                time.sleep(delay)
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
            LAST_REQUEST_AT = time.monotonic()
            return payload
        except (OSError, urllib.error.HTTPError) as error:
            if attempt == 5:
                raise
            time.sleep(10 if getattr(error, "code", None) == 429 else 2 ** attempt)
    raise AssertionError("unreachable")


def wikidata_entities(ids):
    result = {}
    for batch in chunks(ids, 50):
        payload = get_json(WIKIDATA_API, {"action": "wbgetentities", "format": "json", "ids": "|".join(batch),
            "props": "labels|descriptions|sitelinks/urls", "languages": "ja|en", "sitefilter": "jawiki|enwiki"})
        result.update(payload["entities"])
    return result


def plain_wikitext(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text, flags=re.S | re.I)
    for _ in range(12):
        newer = re.sub(r"\{\{[^{}]*\}\}", "", text, flags=re.S)
        if newer == text:
            break
        text = newer
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>|={2,}.*?={2,}|\[https?://[^ ]+ ([^\]]+)\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def japanese_pages(titles, cache_dir):
    result = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    # The revision API can return the entire Wikitext.  Fetching several
    # large pages in one response regularly times out, so request one page at
    # a time and truncate only after plain-text conversion.
    for index, batch in enumerate(chunks(titles, 1), 1):
        print(f"Fetching Japanese article {index}/{len(titles)}", flush=True)
        cache_path = cache_dir / f"{hashlib.sha256(batch[0].encode()).hexdigest()}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            result[cached["title"]] = (cached["text"], cached["revision"])
            continue
        payload = get_json(JAWIKI_API, {"action": "query", "format": "json", "formatversion": "2",
            "prop": "revisions", "titles": "|".join(batch), "rvprop": "ids|timestamp|content", "rvslots": "main", "redirects": "1"})
        for page in payload.get("query", {}).get("pages", []):
            revision = (page.get("revisions") or [{}])[0]
            content = revision.get("slots", {}).get("main", {}).get("content", "")
            if "missing" not in page and content:
                text = plain_wikitext(content)
                cache_path.write_text(json.dumps({"title": page["title"], "text": text, "revision": revision}, ensure_ascii=False))
                result[page["title"]] = (text, revision)
    return result


def language_entity(item, language, site):
    label = item.get("labels", {}).get(language, {}).get("value")
    link = item.get("sitelinks", {}).get(site)
    return label and link


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, help="Optional deterministic limit; default uses all bilingual test triples")
    parser.add_argument("--article-character-limit", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    if args.count is not None and args.count <= 0 or args.article_character_limit < 0:
        raise ValueError("--count and --article-character-limit must be non-negative")

    test_path = CODEX_ROOT / "data" / "triples" / "codex-s" / "test.txt"
    source = test_path.read_bytes()
    triples = [tuple(line.split("\t")) for line in source.decode().splitlines() if line]
    metadata = wikidata_entities(sorted({item for triple in triples for item in triple}))
    extract_zip = CODEX_ROOT / "data" / "entities" / "en" / "extracts.zip"
    with zipfile.ZipFile(extract_zip) as archive:
        extract_names = set(archive.namelist())
        eligible = []
        for triple in triples:
            head, relation, tail = triple
            if all(language_entity(metadata[item], "ja", "jawiki") and language_entity(metadata[item], "en", "enwiki") for item in (head, tail)):
                if f"extracts/{head}.txt" in extract_names and f"extracts/{tail}.txt" in extract_names:
                    eligible.append(triple)
        if args.count:
            eligible = sorted(eligible, key=lambda row: hashlib.sha256("\t".join(row).encode()).hexdigest())[:args.count]
        # A single subject can occur in several test triples.  Article text is
        # keyed by title, so fetch each Japanese article once and reuse it.
        ja_titles = list(dict.fromkeys(metadata[head]["sitelinks"]["jawiki"]["title"] for head, _, _ in eligible))
        ja_pages = japanese_pages(ja_titles, Path("/private/tmp/jev-ja-wikipedia-cache"))
        records = []
        for head, relation, tail in eligible:
            ja_site = metadata[head]["sitelinks"]["jawiki"]
            ja_page = ja_pages.get(ja_site["title"])
            if not ja_page:
                continue
            ja_text, revision = ja_page
            en_text = archive.read(f"extracts/{head}.txt").decode(errors="replace").strip()
            if len(ja_text) < 120 or len(en_text) < 120:
                continue
            def entity(entity):
                return {language: {"label": metadata[entity]["labels"][language]["value"], "wikipedia_url": metadata[entity]["sitelinks"][site]["url"]}
                        for language, site in (("ja", "jawiki"), ("en", "enwiki"))} | {"id": entity}
            records.append({
                "id": f"codex-s-bilingual-test-{len(records)+1:04d}", "task": "relation_choice",
                "source": {"dataset": "CoDEx-S", "split": "test", "triple": [head, relation, tail]},
                "subject": entity(head), "object": entity(tail),
                "articles": {
                    "ja": {"title": ja_site["title"], "revision_id": revision.get("revid"), "revision_timestamp": revision.get("timestamp"),
                           "revision_url": f"https://ja.wikipedia.org/w/index.php?oldid={revision.get('revid')}", "text": ja_text[:args.article_character_limit] if args.article_character_limit else ja_text},
                    "en": {"title": metadata[head]["sitelinks"]["enwiki"]["title"], "wikipedia_url": metadata[head]["sitelinks"]["enwiki"]["url"],
                           "source": "CoDEx bundled English Wikipedia extract", "text": en_text[:args.article_character_limit] if args.article_character_limit else en_text},
                },
                "expected_relation": {"id": relation, "labels": {language: metadata[relation]["labels"][language]["value"] for language in ("ja", "en")},
                                      "descriptions": {language: metadata[relation].get("descriptions", {}).get(language, {}).get("value", "") for language in ("ja", "en")}},
            })
    relation_ids = sorted({row["expected_relation"]["id"] for row in records})
    relations = [{"id": relation, "labels": {language: metadata[relation]["labels"][language]["value"] for language in ("ja", "en")},
                  "descriptions": {language: metadata[relation].get("descriptions", {}).get(language, {}).get("value", "") for language in ("ja", "en")}} for relation in relation_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    (args.output_dir / "relation_choices.json").write_text(json.dumps({"dataset": "dataset.jsonl", "relations": relations}, ensure_ascii=False, indent=2) + "\n")
    manifest = {"records": len(records), "candidate_bilingual_test_triples": len(eligible), "source_url": TEST_URL,
        "source_sha256": hashlib.sha256(source).hexdigest(), "source_triples": len(triples), "article_character_limit": args.article_character_limit,
        "languages": ["ja", "en"], "english_text_source": "CoDEx bundled English Wikipedia extracts", "relation_choices": len(relations)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(records)} bilingual records with {len(relations)} relation choices")


if __name__ == "__main__":
    main()
