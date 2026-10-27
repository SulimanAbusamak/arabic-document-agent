"""
What a valid invoice record looks like, and the rules it must satisfy.

Two layers, deliberately separate:

  TYPE validation  (Pydantic)
      Is total_amount actually a number? Is issue_date a real date?
      Is the VAT number 15 digits? These are facts about the SHAPE of the
      data. Failing one means the extraction produced something unusable.

  BUSINESS rules   (check_invoice)
      Does subtotal + VAT equal the total? Does the VAT amount match the
      stated rate? These are facts about whether the numbers are CONSISTENT.
      Failing one means a digit was probably misread.

The split matters. A type failure is "I cannot use this at all". A business
rule failure is "this parsed fine but something looks wrong" -- and that is
a signal the agent acts on, not a crash.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

import config


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

class LineItem(BaseModel):
    """
    One row of the items table.

    Every field is optional on purpose. Line items are here for semantic
    matching -- what was actually bought -- not for payment. A row whose
    quantity was illegible must not reject the whole invoice, because the
    totals are what gets paid and they are checked separately.
    """
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    tax_rate: float | None = None
    tax_amount: float | None = None
    line_total: float | None = None


class Invoice(BaseModel):
    """
    A validated invoice. Building one of these either succeeds with usable
    data, or raises ValidationError explaining exactly which field is wrong.
    """

    """
    One row of the items table.

    Every field is optional on purpose. Line items are here for semantic
    matching -- what was actually bought -- not for payment. A row whose
    quantity was illegible must not reject the whole invoice, because the
    totals are what gets paid and they are checked separately.

    tax_rate and tax_amount are per ROW because an invoice can charge
    different rates on different lines. A real one did: 15% on one product
    and 10% on another, which the single invoice-level vat_rate could not
    represent, so validation called a correct document an arithmetic error.
    """
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    tax_rate: float | None = None
    tax_amount: float | None = None
    line_total: float | None = None

    vendor_name: str = Field(min_length=2)
    vendor_name_key: str | None = None        # normalised, for matching
    vendor_vat_number: str | None = None

    buyer_name: str | None = None
    buyer_name_key: str | None = None
    buyer_vat_number: str | None = None

    invoice_number: str = Field(min_length=1)
    issue_date: date | None = None

    subtotal: float = Field(ge=0)             # ge = greater than or equal
    vat_rate: float = Field(ge=0, le=1)       # 0.15, not 15
    vat_amount: float = Field(ge=0)
    total_amount: float = Field(gt=0)         # gt = strictly greater than

    currency: str = Field(default="SAR", min_length=3, max_length=3)
    discount: float | None = None      # applied before tax, if any
    line_items: list[LineItem] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    unreadable_fields: list[str] = Field(default_factory=list)

    @field_validator("vendor_vat_number", "buyer_vat_number")
    @classmethod
    def clean_vat_number(cls, v: str | None) -> str | None:
        """
        Keep the digits. Store whatever came through.

        This used to REJECT anything that was not 15 digits, which was the
        wrong place for that check. A malformed VAT number does not make the
        record unusable -- the amounts, dates and names read fine -- and
        rejecting here meant the arithmetic rules never ran at all. dev_022
        had three problems and could only ever report one of them.

        Format is a BUSINESS rule, so it moved to check_invoice(), where it
        becomes a finding the agent weighs instead of a fatal error.
        """
        if v is None or v == "":
            return None
        return re.sub(r"\D", "", v) or None

    @field_validator("currency", mode="before")
    @classmethod
    def default_currency(cls, v) -> str:
        """
        Fall back to SAR when the document does not print a currency.

        Till receipts usually do not: 'Total Cash 128.00' and nothing else.
        The v4 prompt hid this by telling the model to answer SAR; v5 asks
        it to read what is printed, so an honest null now arrives here --
        and a missing currency should not make an otherwise perfect receipt
        unusable.

        SAR is the right assumption for this system, and it is an
        assumption: an invoice in dollars that omits the symbol would be
        silently mislabelled.
        """
        if v is None or not str(v).strip():
            return "SAR"
        return str(v).upper()

    @property
    def vat_convention(self) -> str:
        """
        'inclusive' if the printed prices already contain the VAT.

        Saudi retail works this way: the shelf price is what you pay, and
        the VAT is the portion already inside it. ZATCA calls these
        documents simplified tax invoices (فاتورة ضريبية مبسطة), and they
        are a whole legitimate invoice type -- every restaurant receipt an
        employee submits for reimbursement is one.

        None of the generated invoices used it, because generate.py always
        builds subtotal -> add VAT -> total. So the convention was invisible
        until a real receipt arrived, and the validation called it an
        arithmetic error.

        The signal: VAT is non-zero but the total equals the subtotal. Under
        exclusive pricing that is a contradiction -- adding VAT has to make
        the total bigger. Under inclusive pricing it is just how the
        document is written.
        """
        if self.vat_rate <= 0:
            return "exclusive"
        tol = config.MONEY_TOLERANCE
        if abs(self.total_amount - self.subtotal) <= tol and self.vat_amount > tol:
            return "inclusive"
        return "exclusive"


# --------------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------------
Severity = Literal["error", "warning"]


class Issue(BaseModel):
    """One thing that looks wrong, and how badly."""
    rule: str
    severity: Severity
    detail: str


def check_invoice(inv: Invoice) -> list[Issue]:
    """
    Run the consistency checks. Returns a list of problems -- empty means
    everything adds up.

    Nothing here raises. These are findings for the agent to weigh, not
    reasons to stop.
    """
    issues: list[Issue] = []
    tol = config.MONEY_TOLERANCE

    # --- the money has to hang together ----------------------------------
    # Which arithmetic applies depends on the invoice's pricing convention,
    # and the document never says which one it uses. See Invoice.vat_convention.
    if inv.vat_convention == "inclusive":
        # The total is what was paid; the VAT is the slice already inside it.
        #     net = total / 1.15        VAT = total - net
        # "subtotal + VAT = total" is not checked here, because under this
        # convention the subtotal IS the total -- that is what identified it.
        net = round(inv.total_amount / (1 + inv.vat_rate), 2)
        expected_vat = round(inv.total_amount - net, 2)
        if abs(expected_vat - inv.vat_amount) > tol:
            issues.append(Issue(
                rule="vat_inside_total",
                severity="error",
                detail=(f"VAT-inclusive invoice: {inv.total_amount} contains "
                        f"{expected_vat} of VAT at {inv.vat_rate:.0%}, "
                        f"but the invoice says {inv.vat_amount}"),
            ))
    else:
        # --- the arithmetic must close ------------------------------------
        expected_total = round(inv.subtotal + inv.vat_amount, 2)
        if abs(expected_total - inv.total_amount) > tol:
            issues.append(Issue(
                rule="total_equals_subtotal_plus_vat",
                severity="error",
                detail=(f"{inv.subtotal} + {inv.vat_amount} = {expected_total}, "
                        f"but total says {inv.total_amount}"),
            ))

        # --- VAT must match what the rows say, or the printed rate -------
        # Per-row first, when every row carries its own tax amount: that is
        # the only way to validate an invoice with more than one rate on it.
        #
        # All or nothing, deliberately. If three rows out of four have a tax
        # amount, summing the three is wrong by whatever the fourth was --
        # and filling the gap would mean guessing that row's rate. Partial
        # data is exactly where not to improvise, so we fall back to the
        # invoice totals, which are figures we definitely have.
        row_tax = [i.tax_amount for i in inv.line_items
                   if i.tax_amount is not None]
        if inv.line_items and len(row_tax) == len(inv.line_items):
            expected_vat = round(sum(row_tax), 2)
            rule_name = "vat_matches_line_items"
            how = f"the {len(row_tax)} line item tax amounts sum to"
        else:
            expected_vat = round(inv.subtotal * inv.vat_rate, 2)
            rule_name = "vat_matches_rate"
            how = f"{inv.subtotal} x {inv.vat_rate:.0%} ="

        if abs(expected_vat - inv.vat_amount) > tol * max(1, len(row_tax)):
            issues.append(Issue(
                rule=rule_name,
                severity="error",
                detail=(f"{how} {expected_vat}, "
                        f"but the invoice says {inv.vat_amount}"),
            ))

    # --- the rate itself should be one we recognise -----------------------
    if inv.vat_rate not in config.KNOWN_VAT_RATES:
        issues.append(Issue(
            rule="unusual_vat_rate",
            severity="warning",
            detail=(f"rate {inv.vat_rate:.1%} is not a known rate "
                    f"({', '.join(f'{r:.0%}' for r in sorted(config.KNOWN_VAT_RATES))})"),
        ))
    # --- VAT registration numbers must be the right shape ------------------
    for label, number in (("supplier", inv.vendor_vat_number),
                          ("buyer", inv.buyer_vat_number)):
        if number is None:
            continue
        if len(number) != config.VAT_NUMBER_LENGTH:
            issues.append(Issue(
                rule=f"{label}_vat_number_length",
                severity="error",
                detail=(f"{label} VAT number must be "
                        f"{config.VAT_NUMBER_LENGTH} digits, got "
                        f"{len(number)}: {number}"),
            ))
        elif not (number.startswith("3") and number.endswith("3")):
            issues.append(Issue(
                rule=f"{label}_vat_number_format",
                severity="error",
                detail=(f"{label} VAT number must start and end with 3: "
                        f"{number}"),
            ))

    # --- missing fields ---------------------------------------------------
    if inv.vendor_vat_number is None:
        issues.append(Issue(
            rule="missing_vendor_vat",
            severity="warning",
            detail="no supplier VAT number -- required on a Saudi tax invoice",
        ))

    if inv.issue_date is None:
        issues.append(Issue(
            rule="missing_issue_date",
            severity="warning",
            detail="could not read an issue date",
        ))

    # --- the model told us it struggled -----------------------------------
    if inv.unreadable_fields:
        issues.append(Issue(
            rule="unreadable_fields",
            severity="warning",
            detail=f"model could not read: {', '.join(inv.unreadable_fields)}",
        ))

    if inv.confidence < config.CONFIDENCE_FLOOR:
        issues.append(Issue(
            rule="low_confidence",
            severity="error",
            detail=(f"confidence {inv.confidence:.2f} is below the floor "
                    f"of {config.CONFIDENCE_FLOOR}"),
        ))

    return issues


def has_errors(issues: list[Issue]) -> bool:
    """True if anything serious was found."""
    return any(i.severity == "error" for i in issues)
