"""
The whole chain in one call: image in, validated result out.

    python src/pipeline.py                 # one dev invoice
    python src/pipeline.py dev_004.png     # a specific one
    python src/pipeline.py dev_004.png --no-retry

    from pipeline import process
    result = process(path)

This is the function everything else builds on -- the agent on Day 4, the
evaluation on Day 5, the API on Day 6. They all call process() and work
with what it returns.

Design rule: process() NEVER raises on a bad invoice.

A blurry scan, a misread digit, a vendor name Claude could not make out --
none of those are exceptional. They are the normal variety of real post.
The caller needs to know WHAT went wrong so it can route the document, and
an exception would just kill the batch at document 17.

So every outcome comes back as the same shape, with `status` saying which
of the three things happened.

A SECOND LOOK
-------------
A document that fails the first read is not given up on. It goes back
through with a stronger model -- a cheap-first cascade: Haiku reads all 50,
Sonnet only ever sees the handful Haiku could not manage.

Enlarging the image was the other obvious lever and it is not available:
these models look at images up to 2576 px on the long edge, and our
invoices are already 2246, so anything bigger is downscaled by the API
before the model sees it. The pixels we have are all the pixels there are.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

import config
from extract import extract
from models import Invoice, Issue, check_invoice, has_errors
from normalize import normalize_record

Status = Literal["ok", "flagged", "unusable"]


@dataclass
class Result:
    """
    What came of one document.

    status:
      "ok"        parsed, validated, nothing wrong -> safe to act on
      "flagged"   parsed, but a check failed       -> a human should look
      "unusable"  could not even build a record    -> reject it
    """
    image: str
    status: Status
    invoice: Invoice | None = None
    issues: list[Issue] = field(default_factory=list)
    raw_fields: dict = field(default_factory=dict)   # what Claude said, untouched
    error: str | None = None                         # why it was unusable
    meta: dict = field(default_factory=dict)         # model, tokens, cache hit

    # --- the second look -------------------------------------------------
    retried: bool = False          # a stronger model was asked as well
    recovered: bool = False        # ...and it did better than the first
    first_status: Status | None = None   # what the cheap model produced
    first_error: str | None = None       # and why, if it failed

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> str:
        """One line, for logs and terminal output."""
        tail = ""
        if self.recovered:
            tail = f"  [recovered from {self.first_status} by {self.meta.get('model')}]"
        elif self.retried:
            tail = "  [retried, no better]"

        if self.invoice is None:
            return f"{self.image}: UNUSABLE - {self.error}{tail}"
        inv = self.invoice
        head = (f"{self.image}: {self.status.upper()} "
                f"{inv.total_amount:,.2f} {inv.currency} from {inv.vendor_name}")
        if self.issues:
            head += f"  [{len(self.issues)} issue(s)]"
        return head + tail


# --------------------------------------------------------------------------
# One pass
# --------------------------------------------------------------------------
def attempt(image_path: Path, model: str | None = None,
            use_cache: bool = True) -> Result:
    """
    One read of one document: extract -> normalise -> validate -> Result.

    Pulled out of process() so it can be run twice with different models
    without duplicating any of the chain.
    """
    name = image_path.name

    # --- 1. read the document --------------------------------------------
    try:
        extracted = extract(image_path, model=model, use_cache=use_cache)
    except Exception as e:
        # network trouble, malformed JSON, API error
        return Result(image=name, status="unusable",
                      error=f"extraction failed: {type(e).__name__}: {e}",
                      meta={"model": model or config.MODEL})

    raw = extracted["fields"]
    meta = dict(extracted["_meta"])
    meta["from_cache"] = extracted.get("_from_cache", False)

    # --- 2. clean it up ---------------------------------------------------
    try:
        cleaned = normalize_record(raw)
    except Exception as e:
        return Result(image=name, status="unusable", raw_fields=raw, meta=meta,
                      error=f"normalisation failed: {type(e).__name__}: {e}")

    # An invoice that prints no rate still needs one for the arithmetic check.
    if cleaned.get("vat_rate") is None:
        cleaned["vat_rate"] = config.DEFAULT_VAT_RATE

    # --- 3. is it even a usable record? -----------------------------------
    try:
        invoice = Invoice(**{k: v for k, v in cleaned.items()
                             if k in Invoice.model_fields})
    except ValidationError as e:
        # Report every bad field at once, not just the first.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        return Result(image=name, status="unusable", raw_fields=raw, meta=meta,
                      error=f"validation failed: {problems}")

    # --- 4. do the numbers hang together? ---------------------------------
    issues = check_invoice(invoice)
    status: Status = "flagged" if issues else "ok"

    return Result(image=name, status=status, invoice=invoice,
                  issues=issues, raw_fields=raw, meta=meta)


# --------------------------------------------------------------------------
# Two passes, when the first one is not good enough
# --------------------------------------------------------------------------
def needs_retry(result: Result) -> bool:
    """
    Is this worth a second, more expensive look?

    Two cases, and the second one is the interesting one:

      unusable          nothing came back we could use. Obvious.

      an error-severity  the record parsed, but the numbers contradict each
      finding            other -- which means a digit was misread. This is
                         the case that matters: dev_022 is 'flagged', not
                         'unusable', and its subtotal + VAT does not equal
                         its total because 12,393.81 was read as 13,393.81.
                         A different model may simply read it correctly.

    Warnings do not qualify. A missing supply date is not a misread digit,
    and paying 20x for a second opinion on it buys nothing.
    """
    return result.status == "unusable" or has_errors(result.issues)


def better(new: Result, old: Result) -> bool:
    """
    Did the second read actually improve on the first?

    Ranked bluntly: a clean record beats a flagged one, and a flagged one
    beats nothing at all. If the retry is no better, the first result is
    kept -- a more expensive model is not automatically a more correct one,
    and silently preferring it would hide that.
    """
    rank = {"unusable": 0, "flagged": 1, "ok": 2}
    if rank[new.status] != rank[old.status]:
        return rank[new.status] > rank[old.status]
    # Same status: fewer hard errors wins.
    return (sum(1 for i in new.issues if i.severity == "error")
            < sum(1 for i in old.issues if i.severity == "error"))


def process(image_path: Path, model: str | None = None,
            use_cache: bool = True, retry: bool = True) -> Result:
    """
    Read a document, and if the cheap model could not manage it, ask a
    stronger one before giving up.

    Costs follow the same shape as the failures: nearly every document is
    settled by the first, cheap read. Only the few that fail cost more.
    """
    first = attempt(image_path, model=model, use_cache=use_cache)

    if not retry or not needs_retry(first):
        return first

    # Do not retry with the model that just failed.
    retry_model = config.RETRY_MODEL
    if (first.meta.get("model") or config.MODEL) == retry_model:
        return first

    second = attempt(image_path, model=retry_model, use_cache=use_cache)

    keep = second if better(second, first) else first
    keep.retried = True
    keep.recovered = keep is second
    keep.first_status = first.status
    keep.first_error = first.error
    return keep


# --------------------------------------------------------------------------
def find_image(name: str) -> Path:
    """Look for an image in dev, then test, then real."""
    for folder in (config.DEV_DIR, config.TEST_DIR, config.REAL_DIR):
        path = folder / name
        if path.exists():
            return path
    raise SystemExit(f"{name} not found in dev, test or real")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "dev_000.png"
    result = process(find_image(name), retry="--no-retry" not in sys.argv)

    print(result.summary())
    print()

    if result.retried:
        print(f"first read ({config.MODEL}): {result.first_status}")
        if result.first_error:
            print(f"  {result.first_error}")
        print(f"second read ({config.RETRY_MODEL}): "
              f"{'kept' if result.recovered else 'no better, discarded'}")
        print()

    if result.invoice:
        inv = result.invoice
        print("validated record:")
        for key in ("vendor_name", "vendor_vat_number", "buyer_name",
                    "invoice_number", "issue_date", "subtotal", "vat_rate",
                    "vat_amount", "total_amount", "currency", "confidence"):
            print(f"  {key:20} {getattr(inv, key)}")

    if result.issues:
        print("\nissues:")
        for i in result.issues:
            print(f"  [{i.severity}] {i.rule}: {i.detail}")

    if result.error:
        print(f"\nerror: {result.error}")
        print("\nwhat Claude returned:")
        print(json.dumps(result.raw_fields, ensure_ascii=False, indent=2))

    m = result.meta
    if m:
        where = "cache" if m.get("from_cache") else "API"
        cost = m.get("cost")
        line = f"\n({m.get('model')} via {where}, prompt {m.get('prompt_version')}"
        if cost:
            line += f", ${cost:.5f}"
        print(line + ")")


if __name__ == "__main__":
    main()
