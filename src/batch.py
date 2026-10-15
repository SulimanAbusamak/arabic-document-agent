"""
Run the pipeline over every invoice and report how it went.

    python src/batch.py              # everything (dev + test)
    python src/batch.py dev          # dev only
    python src/batch.py test         # test only
    python src/batch.py --no-cache   # force fresh API calls (costs money)

Anything already cached is reused, so re-running this is free. Only
documents not yet seen at the current prompt version cost anything.

Results are written to data/batch_results.json so the next steps can read
them without recomputing. Printed output is for you; the file is for the
rest of the pipeline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import config
from pipeline import Result, process

OUTPUT = config.DATA_DIR / "batch_results.json"

# Haiku pricing, per million tokens
PRICE_IN = 0.10
PRICE_OUT = 0.50


def collect_images(which: str) -> list[Path]:
    """Every .png in the chosen split(s), sorted so runs are comparable."""
    folders = {"dev": [config.DEV_DIR],
               "test": [config.TEST_DIR],
               "all": [config.DEV_DIR, config.TEST_DIR]}[which]
    images: list[Path] = []
    for folder in folders:
        images.extend(sorted(folder.glob("*.png")))
    return images


def result_to_dict(r: Result) -> dict:
    """Flatten a Result so it can go in a JSON file."""
    return {
        "image": r.image,
        "status": r.status,
        "error": r.error,
        "invoice": r.invoice.model_dump(mode="json") if r.invoice else None,
        "issues": [i.model_dump() for i in r.issues],
        "raw_fields": r.raw_fields,
        "meta": r.meta,
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0] if args else "all"
    use_cache = "--no-cache" not in sys.argv

    images = collect_images(which)
    if not images:
        raise SystemExit(f"no images found for '{which}'")

    print(f"{len(images)} images, cache {'on' if use_cache else 'OFF'}\n")

    results: list[Result] = []
    tokens_in = tokens_out = 0
    api_calls = 0
    started = time.time()

    for n, path in enumerate(images, start=1):
        result = process(path, use_cache=use_cache)
        results.append(result)

        meta = result.meta
        if meta and not meta.get("from_cache"):
            api_calls += 1
            tokens_in += meta.get("input_tokens", 0)
            tokens_out += meta.get("output_tokens", 0)
            where = "api"
        else:
            where = "   "

        mark = {"ok": "ok      ", "flagged": "FLAGGED ",
                "unusable": "UNUSABLE"}[result.status]
        print(f"  [{n:>2}/{len(images)}] {where} {mark} {path.name}")

    elapsed = time.time() - started

    # ---- summary ---------------------------------------------------------
    counts = {s: sum(1 for r in results if r.status == s)
              for s in ("ok", "flagged", "unusable")}
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1_000_000

    print(f"\n{'-' * 46}")
    for status, n in counts.items():
        pct = 100 * n / len(results)
        print(f"  {status:9} {n:>3}  ({pct:4.1f}%)")
    print(f"{'-' * 46}")
    print(f"  api calls {api_calls:>3}   cached {len(results) - api_calls:>3}")
    print(f"  tokens    in {tokens_in:,}  out {tokens_out:,}")
    print(f"  cost      ${cost:.4f}")
    print(f"  time      {elapsed:.1f}s")

    # ---- what went wrong -------------------------------------------------
    problems = [r for r in results if r.status != "ok"]
    if problems:
        print(f"\nneeding attention ({len(problems)}):")
        for r in problems:
            print(f"  {r.image}")
            if r.error:
                print(f"      {r.error}")
            for i in r.issues:
                print(f"      [{i.severity}] {i.rule}: {i.detail}")

    # ---- save ------------------------------------------------------------
    OUTPUT.write_text(
        json.dumps([result_to_dict(r) for r in results],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
