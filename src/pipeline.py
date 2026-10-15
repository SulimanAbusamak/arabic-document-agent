"""
The whole chain in one call: image in, validated result out.

    python src/pipeline.py                 # one dev invoice
    python src/pipeline.py dev_004.png     # a specific one

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

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> str:
        """One line, for logs and terminal output."""
        if self.invoice is None:
            return f"{self.image}: UNUSABLE - {self.error}"
        inv = self.invoice
        head = (f"{self.image}: {self.status.upper()} "
                f"{inv.total_amount:,.2f} {inv.currency} from {inv.vendor_name}")
        if self.issues:
            head += f"  [{len(self.issues)} issue(s)]"
        return head


def process(image_path: Path, model: str | None = None,
            use_cache: bool = True) -> Result:
    """
    image -> extract -> normalise -> validate -> Result
    """
    name = image_path.name

    # --- 1. read the document --------------------------------------------
    try:
        extracted = extract(image_path, model=model, use_cache=use_cache)
    except Exception as e:
        # network trouble, malformed JSON, API error
        return Result(image=name, status="unusable",
                      error=f"extraction failed: {type(e).__name__}: {e}")

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
def find_image(name: str) -> Path:
    """Look for an image in dev, then test."""
    for folder in (config.DEV_DIR, config.TEST_DIR):
        path = folder / name
        if path.exists():
            return path
    raise SystemExit(f"{name} not found in {config.DEV_DIR} or {config.TEST_DIR}")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "dev_000.png"
    result = process(find_image(name))

    print(result.summary())
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
        print(f"\n({m.get('model')} via {where}, prompt {m.get('prompt_version')})")


if __name__ == "__main__":
    main()
