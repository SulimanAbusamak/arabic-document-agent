"""
Measure how well the whole thing actually works.

    python src/evaluate.py dev        # iterate on these
    python src/evaluate.py test       # ONCE, at the end
    python src/evaluate.py all

Costs nothing -- everything is read from batch_results.json and the
database. No API calls.

Two different numbers get reported, and keeping them apart matters:

  MODEL ACCURACY     what Claude read off the image, before validation
                     had any say. This measures the extraction.

  PIPELINE ACCURACY  what came out the far end as a usable field. A
                     document we rejected scores zero here even if the
                     model read it perfectly, because nothing usable
                     reached the database.

The gap between them is the cost of the refusals. A system that refuses
everything has perfect model accuracy and useless pipeline accuracy.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import config
from database import connect
from normalize import normalize_arabic, normalize_record

RESULTS = config.DATA_DIR / "batch_results.json"
OUTPUT = config.DATA_DIR / "eval_results.json"


# --------------------------------------------------------------------------
# What counts as correct
# --------------------------------------------------------------------------
# Written down explicitly, because "correct" is a judgement call and an
# evaluation whose rules are vague can be argued into any number you like.

def money_ok(truth, got) -> bool:
    """Within the same tolerance the validator uses. 0.02 SAR."""
    if got is None:
        return False
    try:
        return abs(float(truth) - float(got)) <= config.MONEY_TOLERANCE
    except (TypeError, ValueError):
        return False


def text_ok(truth, got) -> bool:
    """
    Names compare after Arabic normalisation.

    مؤسسه النور instead of مؤسسة النور is not a reading error -- the two
    spellings are the same name, and the normaliser exists precisely so
    they match. Counting that as wrong would punish the system for a
    problem it already solved.
    """
    if got is None:
        return False
    return normalize_arabic(str(truth)) == normalize_arabic(str(got))


def exact_ok(truth, got) -> bool:
    """IDs and dates: exact string match, no mercy."""
    if got is None:
        return False
    return str(truth).strip() == str(got).strip()


# field name -> comparator. supply_date and line_item_count are in ground
# truth but not in the Invoice model, so they are not scored.
FIELDS = {
    "vendor_name": text_ok,
    "vendor_vat_number": exact_ok,
    "buyer_name": text_ok,
    "buyer_vat_number": exact_ok,
    "invoice_number": exact_ok,
    "issue_date": exact_ok,
    "subtotal": money_ok,
    "vat_amount": money_ok,
    "total_amount": money_ok,
    "currency": exact_ok,
}

# The fields that actually matter for paying an invoice. A wrong name is
# bad; a wrong total is money out the door.
MONEY_FIELDS = ("subtotal", "vat_amount", "total_amount")

# The minimum a document must give up before it can be posted at all:
# who to pay, which invoice, how much. Get any of these wrong and the
# record is unusable no matter how much else read correctly.
#
# The first version of this file judged refusals on the money fields
# alone, and scored a correct refusal as a mistake: dev_001_deg had all
# three amounts right and no readable supplier name. The amounts are
# worthless if you do not know who to send them to.
POSTABLE_FIELDS = ("vendor_name", "invoice_number", "total_amount")


def postable(scores: dict[str, bool]) -> bool:
    """Could this document be posted to the ledger as read?"""
    return all(scores.get(f, False) for f in POSTABLE_FIELDS)


def compare(truth: dict, got: dict | None) -> dict[str, bool]:
    """
    Score one record field by field.

    got=None means the document was rejected, so every field is a miss.
    That is deliberate: a null is not a free pass. A system that returned
    null for everything would otherwise never be wrong about anything.
    """
    if got is None:
        return {f: False for f in FIELDS}
    return {f: fn(truth[f], got.get(f))
            for f, fn in FIELDS.items() if f in truth}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load() -> tuple[dict, list[dict]]:
    if not config.GROUND_TRUTH.exists():
        raise SystemExit("ground_truth.json missing -- run src/generate.py")
    if not RESULTS.exists():
        raise SystemExit("batch_results.json missing -- run src/batch.py")

    truth = json.loads(config.GROUND_TRUTH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    return truth, results


def score_all(truth: dict, results: list[dict], split: str) -> list[dict]:
    """One scored row per document."""
    rows = []
    for r in results:
        image = r["image"]
        t = truth.get(image)
        if t is None:
            print(f"  !! no ground truth for {image} -- skipped")
            continue
        if split != "all" and t["_meta"].get("split") != split:
            continue

        # What the pipeline produced, validated and normalised.
        pipeline_fields = r.get("invoice")

        # What the model read, regardless of whether we accepted it. The
        # raw extraction goes through the normaliser so the comparison is
        # like for like -- otherwise '19,270.50 SAR' would never match
        # 19270.5 and the model would look far worse than it is.
        raw = r.get("raw_fields") or {}
        model_fields = normalize_record(raw) if raw else None

        rows.append({
            "image": image,
            "status": r["status"],
            "meta": t["_meta"],
            "pipeline": compare(t, pipeline_fields),
            "model": compare(t, model_fields),
            # which model produced the stored read, and whether it took two
            # attempts to get there
            "read_by": (r.get("meta") or {}).get("model"),
            "retried": bool(r.get("retried")),
            "recovered": bool(r.get("recovered")),
            "first_status": r.get("first_status"),
        })
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def field_table(rows: list[dict], key: str, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'field':22} {'correct':>9} {'accuracy':>10}")
    print(f"  {'-' * 22} {'-' * 9} {'-' * 10}")

    totals = correct_total = 0
    for f in FIELDS:
        scored = [r[key][f] for r in rows if f in r[key]]
        if not scored:
            continue
        ok = sum(scored)
        totals += len(scored)
        correct_total += ok
        print(f"  {f:22} {ok:>4}/{len(scored):<4} {100 * ok / len(scored):>9.1f}%")

    if totals:
        print(f"  {'-' * 22} {'-' * 9} {'-' * 10}")
        print(f"  {'ALL FIELDS':22} {correct_total:>4}/{totals:<4} "
              f"{100 * correct_total / totals:>9.1f}%")


def slice_table(rows: list[dict], label: str, bucket) -> None:
    """
    Accuracy broken down by some property of the document.

    One overall number hides everything useful. The whole point of the
    evaluation is finding out WHERE it fails, and that only shows up in
    the slices.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[bucket(r)].append(r)

    print(f"\n  by {label}:")
    print(f"    {'group':24} {'docs':>5} {'fields':>8} {'accuracy':>10}")
    for name in sorted(groups):
        group = groups[name]
        scored = [v for r in group for v in r["model"].values()]
        if not scored:
            continue
        ok = sum(scored)
        print(f"    {name:24} {len(group):>5} {len(scored):>8} "
              f"{100 * ok / len(scored):>9.1f}%")


def retry_report(rows: list[dict]) -> None:
    """
    Did the second opinion earn its money?

    The honest test is not "did the expensive model do better in general"
    but "on the documents the cheap model failed, how many did it rescue,
    and what would those fields have been worth". A cascade that recovers
    nothing is just a slower way to be wrong.
    """
    retried = [r for r in rows if r["retried"]]
    if not retried:
        print("\nno documents were retried")
        return

    recovered = [r for r in retried if r["recovered"]]

    print("\nsecond opinion")
    print(f"  retried           {len(retried)}")
    print(f"  recovered         {len(recovered)}")

    for r in recovered:
        ok = sum(r["pipeline"].values())
        print(f"    {r['image']:20} {r['first_status']} -> {r['status']}"
              f"   {ok}/{len(r['pipeline'])} fields now correct")

    for r in retried:
        if not r["recovered"]:
            print(f"    {r['image']:20} stayed {r['status']} "
                  f"-- both models failed, so the document is the problem")

    # What the retry is worth, in the only unit that matters: correct fields
    # that would otherwise have been wrong.
    gained = sum(sum(r["pipeline"].values()) for r in recovered)
    if gained:
        print(f"\n  {gained} fields reached the database that the first "
              f"read would have lost or got wrong.")


def refusal_analysis(rows: list[dict]) -> None:
    """
    When the system refused a document, was it right to?

    Framed as precision and recall rather than one accuracy number,
    because the two ways of being wrong cost completely different things:

      a needless refusal  -> a human looks at a document that was fine.
                             Wasted five minutes.
      a missed refusal    -> a wrong number went into the ledger.
                             Money.

    You would happily trade several needless refusals for one fewer bad
    number, and a single accuracy figure hides that trade entirely.
    """
    tp = fp = fn = tn = 0
    needless, missed = [], []

    for r in rows:
        refused = r["status"] == "unusable"
        unpostable = not postable(r["model"])

        if refused and unpostable:
            tp += 1
        elif refused and not unpostable:
            fp += 1
            needless.append(r["image"])
        elif not refused and unpostable:
            fn += 1
            missed.append((r["image"], [f for f in POSTABLE_FIELDS
                                        if not r["model"].get(f, False)]))
        else:
            tn += 1

    print("\nrefusal behaviour")
    print("  'unpostable' = supplier, invoice number or total read wrong")
    print(f"\n  {'':28} {'unpostable':>12} {'postable':>12}")
    print(f"  {'refused (unusable)':28} {tp:>12} {fp:>12}")
    print(f"  {'accepted (ok/flagged)':28} {fn:>12} {tn:>12}")

    if tp + fp:
        print(f"\n  precision {tp / (tp + fp):.2f}  "
              f"-- of what we refused, this share was genuinely unusable")
    if tp + fn:
        print(f"  recall    {tp / (tp + fn):.2f}  "
              f"-- of the genuinely unusable, this share was caught")

    if needless:
        print(f"\n  refused although readable ({len(needless)}):")
        for i in needless:
            print(f"    {i}")
    if missed:
        print(f"\n  accepted although unusable ({len(missed)}):")
        for image, bad in missed:
            print(f"    {image:20} wrong: {', '.join(bad)}")

    # --- a second, independent check -------------------------------------
    # degrade.py recorded which documents it expected to be unreadable when
    # it wrecked them. Comparing against that is a different question from
    # the one above: not "was the refusal justified by the data" but "did
    # the system behave the way we designed it to".
    #
    # Where the two disagree, it is worth knowing which one was wrong. A
    # document we expected to be illegible and which read perfectly means
    # our expectation was too pessimistic, not that the system misbehaved.
    expected = [r for r in rows if r["meta"].get("expect_refusal")]
    if expected:
        print(f"\n  documents degraded with refusal expected "
              f"({len(expected)}):")
        for r in expected:
            refused = r["status"] == "unusable"
            verdict = ("refused, as expected" if refused
                       else "read anyway -- "
                            + ("correctly, so the expectation was wrong"
                               if postable(r["model"])
                               else "and it was NOT postable"))
            print(f"    {r['image']:20} {r['status']:9} {verdict}")


def decision_safety(rows: list[dict]) -> None:
    """
    The question that actually matters: did we approve anything wrong?

    Every other number here is diagnostic. This one is the failure mode
    that costs real money, so it gets reported on its own.
    """
    conn = connect()
    latest = {
        r["image"]: r["decision"]
        for r in conn.execute("""
            SELECT d.image, d.decision FROM decisions d
            JOIN (SELECT image, MAX(id) AS id FROM decisions GROUP BY image) m
              ON m.id = d.id
        """)
    }
    conn.close()

    if not latest:
        print("\nno decisions on record -- run src/review.py")
        return

    bad_money, bad_other, approved = [], [], 0
    for r in rows:
        if latest.get(r["image"]) != "approve":
            continue
        approved += 1
        if not all(r["model"].get(f, False) for f in MONEY_FIELDS):
            bad_money.append(r["image"])
        elif not postable(r["model"]):
            bad_other.append(r["image"])

    print(f"\ndecision safety")
    print(f"  approved                        {approved}")
    print(f"  approved with a wrong amount    {len(bad_money)}")
    print(f"  approved but not postable       {len(bad_other)}")
    for i in bad_money + bad_other:
        print(f"    {i}")
    if approved and not bad_money and not bad_other:
        print("\n  nothing with a misread amount or missing key field "
              "was approved.")


# --------------------------------------------------------------------------
def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if split not in {"dev", "test", "all"}:
        raise SystemExit("usage: python src/evaluate.py [dev|test|all]")

    if split in {"test", "all"}:
        print("\n  *** This includes the TEST split. Run it once, at the end,")
        print("  *** and do not tune anything on what you see.\n")

    truth, results = load()
    rows = score_all(truth, results, split)
    if not rows:
        raise SystemExit(f"nothing scored for split '{split}'")

    print(f"\n{'=' * 56}")
    print(f"  {split.upper()}  --  {len(rows)} documents")
    print(f"{'=' * 56}")

    field_table(rows, "model", "MODEL ACCURACY  (what Claude read)")
    field_table(rows, "pipeline",
                "PIPELINE ACCURACY  (what reached the database)")

    print(f"\n{'-' * 56}")
    print("WHERE IT FAILS")
    slice_table(rows, "numerals", lambda r:
                "arabic-indic ٠-٩" if r["meta"].get("arabic_indic_digits")
                else "western 0-9")
    slice_table(rows, "scan quality", lambda r:
                "degraded" if "_deg" in r["image"] else "clean")
    slice_table(rows, "template", lambda r: r["meta"].get("template", "?"))
    slice_table(rows, "date style", lambda r: r["meta"].get("date_style", "?"))
    slice_table(rows, "model that read it", lambda r:
                (r["read_by"] or "?").replace("claude-", ""))

    print(f"\n{'-' * 56}")
    retry_report(rows)
    refusal_analysis(rows)
    decision_safety(rows)

    OUTPUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nsaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
