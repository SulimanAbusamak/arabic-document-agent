"""
Run the pipeline over every invoice and report how it went.

    python src/batch.py              # everything (dev + test)
    python src/batch.py dev          # dev only
    python src/batch.py test         # test only
    python src/batch.py --no-cache   # force fresh API calls (costs money)
    python src/batch.py --no-retry   # first read only, no second opinion

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
from collections import Counter
from pathlib import Path

import config
from pipeline import Result, process

OUTPUT = config.DATA_DIR / "batch_results.json"


def collect_images(which: str) -> list[Path]:
    """Every .png in the chosen split(s), sorted so runs are comparable."""
    folders = {"dev": [config.DEV_DIR],
               "test": [config.TEST_DIR],
               "real": [config.REAL_DIR],
               "all": [config.DEV_DIR, config.TEST_DIR]}[which]
    images: list[Path] = []
    for folder in folders:
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            images.extend(folder.glob(pattern))
    return sorted(images, key=lambda p: p.name)


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
        "retried": r.retried,
        "recovered": r.recovered,
        "first_status": r.first_status,
        "first_error": r.first_error,
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0] if args else "all"

    output = (config.DATA_DIR / "batch_results_real.json" if which == "real" else OUTPUT)

    use_cache = "--no-cache" not in sys.argv
    retry = "--no-retry" not in sys.argv

    images = collect_images(which)
    if not images:
        raise SystemExit(f"no images found for '{which}'")

    print(f"{len(images)} images, cache {'on' if use_cache else 'OFF'}, "
          f"retry {'on' if retry else 'OFF'}\n")

    results: list[Result] = []
    # Cost is summed from each call's own price, not computed at the end from
    # one hardcoded rate. There are two models in play now and they differ by
    # 20x, so a single rate would be wrong -- and wrong in the direction that
    # understates the bill.
    cost = 0.0
    api_calls = 0
    by_model: Counter[str] = Counter()
    started = time.time()

    for n, path in enumerate(images, start=1):
        result = process(path, use_cache=use_cache, retry=retry)
        results.append(result)

        meta = result.meta
        if meta and not meta.get("from_cache"):
            api_calls += 1
            cost += meta.get("cost", 0.0)
            by_model[meta.get("model", "?")] += 1
            where = "api"
        else:
            where = "   "

        mark = {"ok": "ok      ", "flagged": "FLAGGED ",
                "unusable": "UNUSABLE"}[result.status]
        note = ("  <- recovered" if result.recovered
                else "  <- retried, no better" if result.retried else "")
        print(f"  [{n:>2}/{len(images)}] {where} {mark} {path.name}{note}")

    elapsed = time.time() - started

    # ---- summary ---------------------------------------------------------
    counts = {s: sum(1 for r in results if r.status == s)
              for s in ("ok", "flagged", "unusable")}
    retried = [r for r in results if r.retried]
    recovered = [r for r in results if r.recovered]

    print(f"\n{'-' * 52}")
    for status, n in counts.items():
        pct = 100 * n / len(results)
        print(f"  {status:9} {n:>3}  ({pct:4.1f}%)")
    print(f"{'-' * 52}")
    print(f"  api calls {api_calls:>3}")
    for model, n in by_model.most_common():
        print(f"    {model:24} {n:>3}")
    print(f"  cost      ${cost:.4f}")
    print(f"  time      {elapsed:.1f}s")

    # ---- did the second opinion earn its money? --------------------------
    # The only honest way to judge the retry: how many documents it actually
    # rescued, against what the extra calls cost.
    if retried:
        print(f"\n  second opinions   {len(retried)}")
        print(f"  recovered         {len(recovered)}")
        for r in recovered:
            print(f"    {r.image:20} {r.first_status} -> {r.status}")
        no_better = [r for r in retried if not r.recovered]
        if no_better:
            print(f"  no better         {len(no_better)}")
            for r in no_better:
                print(f"    {r.image:20} stayed {r.status}")

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
    output.write_text(
        json.dumps([result_to_dict(r) for r in results],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved -> {output}")


if __name__ == "__main__":
    main()
