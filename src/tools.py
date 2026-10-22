"""
The eight things the agent is allowed to do.

    python src/tools.py          # test every tool by hand, no API calls

Two kinds of tool live here, and the split is deliberate:

  READ   search_vendor, check_duplicate, get_vendor_history,
         find_similar_invoices, find_similar_vendors
         Gather context. Safe to call, call as often as you like, change
         nothing. The agent uses these to answer "is this normal?"

         The last two search by MEANING rather than by exact match, and
         they exist because exact matching is blind to the cases that
         matter: the same job billed twice under a different number, and
         the same company under a different trading name.

  WRITE  approve, escalate, rescan
         End the conversation. Exactly one of these runs per invoice, and
         it is the recorded outcome.

Note what the write tools do NOT do: they do not touch the database. They
return a decision dict and nothing else. The agent loop stores it once the
run is over. That means every tool in this file is a pure-ish function you
can test without an API key and without writing to anything -- which is
why `python src/tools.py` is safe to run as many times as you want.

SCHEMAS is the half Claude reads. The functions are the half Python runs.
They must stay in step: if you add an argument to a function, add it to the
schema too, or the model will never know it exists.
"""

from __future__ import annotations

import json
import sqlite3

from database import connect
from normalize import normalize_arabic

# --------------------------------------------------------------------------
# What Claude sees
# --------------------------------------------------------------------------
# This is a description, not code. Claude reads these names, descriptions
# and argument types and decides which one to ask for. The description text
# is the only instruction it gets about WHEN to use a tool, so it is worth
# writing carefully -- a vague description produces a confused agent.

SCHEMAS = [
    {
        "name": "search_vendor",
        "description": (
            "Look up a supplier we have seen before, by name. Returns their "
            "invoice count, total billed, average invoice value and the date "
            "range we have on file, or known=false if this is a supplier we "
            "have never processed. Matching ignores Arabic spelling variants, "
            "so مؤسسة النور and مؤسسه النور find the same record. All figures "
            "EXCLUDE the invoice you are reviewing, so the average is a "
            "genuine baseline to compare it against. "
            "Call this first on every invoice: whether a supplier is new or "
            "established changes how much scrutiny the invoice deserves."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "The supplier name exactly as it appears "
                                   "on the invoice.",
                },
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "check_duplicate",
        "description": (
            "Check whether this exact invoice has already been processed. "
            "Searches for an existing invoice with the same supplier and the "
            "same invoice number, and reports whether the amount matches too. "
            "The current document is excluded from the search, so a hit means "
            "a genuine second copy. "
            "Call this on EVERY invoice before approving anything. A duplicate "
            "invoice is the most common way an organisation pays the same bill "
            "twice, and the second copy usually looks perfectly valid on its "
            "own -- validation checks cannot catch it, only a database lookup "
            "can. If this returns duplicate=true, escalate; never approve."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "The supplier name as printed on the "
                                   "invoice.",
                },
                "invoice_number": {
                    "type": "string",
                    "description": "The invoice number as printed, e.g. "
                                   "INV-2026-9563.",
                },
            },
            "required": ["vendor_name", "invoice_number"],
        },
    },
    {
        "name": "find_similar_invoices",
        "description": (
            "Search this supplier's invoice history for work described "
            "similarly to the invoice under review, by meaning rather than "
            "by exact wording. Returns matches with a similarity score from "
            "0 to 1, plus each one's invoice number, date and total. "
            "This catches what check_duplicate cannot: the same job billed "
            "twice under a different invoice number and reworded, e.g. "
            "'تركيب شبكات داخلية' and 'أعمال تمديد شبكات'. "
            "IMPORTANT -- a high score alone is not a duplicate. Recurring "
            "work scores high every time it recurs: a monthly maintenance "
            "contract is near-identical every month and is perfectly "
            "legitimate. Judge the PATTERN, using the dates and amounts: "
            "one invoice a month at the same amount is a contract; two in "
            "the same period for the same amount is a double billing. "
            "Call this on invoices over a few thousand riyal, after "
            "check_duplicate comes back clean."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "The supplier name as printed on the "
                                   "invoice.",
                },
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "find_similar_vendors",
        "description": (
            "Check whether a supplier name matches a company already on "
            "file under a different spelling or a different trading name. "
            "Returns the closest names with a similarity score. "
            "Use this when search_vendor reports known=false, before "
            "treating a supplier as new. Exact matching handles spelling "
            "variants like مؤسسه/مؤسسة, but not length variants: "
            "'مؤسسة النور' and 'مؤسسة النور للتوريدات' are one company and "
            "do not match exactly. A score above about 0.85 is very likely "
            "the same company; below 0.5 is a different one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "The supplier name as printed on the "
                                   "invoice.",
                },
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "get_vendor_history",
        "description": (
            "List a supplier's recent invoices: invoice number, date, total "
            "and status, newest first, EXCLUDING the invoice you are "
            "reviewing. Use this when search_vendor shows the supplier is "
            "known and you need to judge whether THIS invoice's amount is "
            "typical for them. An amount several times their usual is worth "
            "escalating even when every validation check passed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "The supplier name as printed on the "
                                   "invoice.",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many invoices to return. Default 10.",
                },
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "approve",
        "description": (
            "Approve this invoice for payment. Final -- use it only when the "
            "validation checks passed, check_duplicate came back clean, the "
            "amount is in keeping with what this supplier normally bills, and "
            "nothing needs a human's eyes. Never approve an invoice carrying "
            "an error-severity issue, and never approve a duplicate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One sentence on why this is safe to pay. "
                                   "Cite what you checked.",
                },
            },
            "required": ["reason"],
        },
    },
    {
        "name": "escalate",
        "description": (
            "Send this invoice to a human reviewer. Final. Use it when the "
            "invoice is readable but something is off: a duplicate, a "
            "validation warning, an amount out of line with the supplier's "
            "history, a supplier we have never seen billing a large sum, or "
            "anything you are unsure about. Escalating is cheap; a wrong "
            "approval is not."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One sentence a reviewer can act on. Say "
                                   "what is wrong and what you compared it "
                                   "against, with the numbers.",
                },
            },
            "required": ["reason"],
        },
    },
    {
        "name": "rescan",
        "description": (
            "Send this document back to be scanned again. Final. Use it when "
            "the document itself cannot be worked with: a required field is "
            "missing or illegible. This is about the IMAGE being unusable, "
            "not about the invoice being wrong -- a readable invoice whose "
            "numbers look suspicious is an escalate, because only a person "
            "holding the original can tell a supplier's mistake from a "
            "misreading on our side. Never accuse the supplier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One sentence: which field could not be "
                                   "read, and enough detail (invoice number, "
                                   "date, amount) for someone to find the "
                                   "original document.",
                },
            },
            "required": ["reason"],
        },
    },
]

TOOL_NAMES = [s["name"] for s in SCHEMAS]
DECISION_TOOLS = {"approve", "escalate", "rescan"}
# No return_to_supplier here, deliberately. That outcome exists, but a human
# sets it after reviewing an escalation -- the agent is never allowed to
# accuse a supplier, because a failed arithmetic check is just as likely to
# be our own misread digit (see the note in agent.py).


# --------------------------------------------------------------------------
# What Python runs
# --------------------------------------------------------------------------
def search_vendor(conn: sqlite3.Connection, vendor_name: str,
                  current_image: str | None = None) -> dict:
    """
    Have we dealt with this supplier before -- APART from this invoice?

    The name arrives as printed on the invoice. Vendor records are keyed on
    the NORMALISED name, so it has to be flattened first -- this is the Day 2
    normaliser earning its keep. Without it, مؤسسه النور would look like a
    brand-new supplier every time the spelling varied.

    The current document is excluded from the figures. That exclusion is the
    whole point of the tool: the question is "what do we know OTHER than what
    this document says". Leave it in and the invoice becomes its own evidence
    -- its amount drags the average toward itself, so nothing ever looks
    unusual, and it shows up in its own history looking like a duplicate.

    Computed straight from the invoices table rather than read off the
    vendors summary, because a precomputed total cannot leave one row out.
    """
    key = normalize_arabic(vendor_name)
    if not key:
        return {"error": "search_vendor needs a vendor_name"}

    row = conn.execute(
        """
        SELECT MIN(vendor_name)  AS display_name,
               COUNT(*)          AS invoice_count,
               SUM(total_amount) AS total_billed,
               MIN(issue_date)   AS first_seen,
               MAX(issue_date)   AS last_seen
        FROM invoices
        WHERE vendor_name_key = ?
          AND image != COALESCE(?, '')
          AND status != 'unusable'
        """,
        (key, current_image),
    ).fetchone()

    count = (row["invoice_count"] if row else 0) or 0
    if count == 0:
        return {
            "known": False,
            "searched_for": vendor_name,
            "note": "No previous invoices from this supplier.",
        }

    # The supplier's VAT number is whichever value appears MOST OFTEN across
    # their invoices -- not MAX, which an earlier version used and which could
    # pick a corrupted read. One misread digit on one document then became the
    # reference everything else was compared against, and the agent reported a
    # mismatch on the documents that were right.
    #
    # A majority vote is the cheapest possible correction: the truth is on most
    # of the documents, so the odd bad read loses.
    vat_row = conn.execute(
        """
        SELECT vendor_vat_number, COUNT(*) AS n
        FROM invoices
        WHERE vendor_name_key = ?
          AND vendor_vat_number IS NOT NULL
          AND image != COALESCE(?, '')
          AND status != 'unusable'
        GROUP BY vendor_vat_number
        ORDER BY n DESC, vendor_vat_number
        LIMIT 1
        """,
        (key, current_image),
    ).fetchone()

    total = row["total_billed"] or 0.0
    return {
        "known": True,
        "display_name": row["display_name"],
        "vat_number": vat_row["vendor_vat_number"] if vat_row else None,
        "vat_number_seen_on": vat_row["n"] if vat_row else 0,
        "invoice_count": count,
        "total_billed": round(total, 2),
        # The number the agent actually reasons with. Handing it the average
        # directly is better than making it divide two numbers itself.
        "average_invoice": round(total / count, 2),
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "note": "Figures exclude the invoice under review. vat_number is the "
                "value appearing on the most invoices from this supplier.",
    }


def check_duplicate(conn: sqlite3.Connection, vendor_name: str,
                    invoice_number: str,
                    current_image: str | None = None) -> dict:
    """
    Has this invoice already been processed?

    Matches on supplier + invoice number, which is what identifies a document
    regardless of how it arrived. Two scans of one invoice, or a vendor who
    submits the same bill twice, both land here.

    `current_image` is excluded from the search. Without that, every invoice
    would find itself and report a duplicate. Note that it is NOT a model
    argument -- the agent loop supplies it, because the model has no business
    knowing or choosing which file it is looking at. Some tool arguments come
    from Claude; some are injected by the code around it.

    Matching on the number alone would be too loose -- invoice numbering is
    per-vendor, so two suppliers can legitimately both have INV-001.
    """
    key = normalize_arabic(vendor_name)
    if not key or not invoice_number:
        return {"error": "check_duplicate needs both vendor_name and "
                         "invoice_number"}

    number = str(invoice_number).strip()
    rows = conn.execute(
        """
        SELECT image, invoice_number, issue_date, total_amount, status
        FROM invoices
        WHERE vendor_name_key = ?
          AND invoice_number = ?
          AND image != COALESCE(?, '')
          AND status != 'unusable'
        ORDER BY issue_date
        """,
        (key, number, current_image),
    ).fetchall()

    if not rows:
        return {
            "duplicate": False,
            "checked": f"{vendor_name} / {number}",
            "note": "No previous invoice with this number from this supplier.",
        }

    return {
        "duplicate": True,
        "match_count": len(rows),
        "matches": [
            {
                "image": r["image"],
                "issue_date": r["issue_date"],
                "total_amount": r["total_amount"],
            }
            for r in rows
        ],
        "note": "This invoice number has already been processed for this "
                "supplier. Paying it again would be a double payment.",
    }


def get_vendor_history(conn: sqlite3.Connection, vendor_name: str,
                       limit: int = 10,
                       current_image: str | None = None) -> dict:
    """
    Recent invoices from one supplier, newest first, excluding this one.

    Same exclusion as search_vendor, for the same reason. Without it the
    invoice under review appears in its own history, where it reads as an
    exact duplicate of itself -- which is how the agent ended up escalating
    dev_000 with a flat contradiction between two of its own tool results.
    """
    key = normalize_arabic(vendor_name)
    rows = conn.execute(
        """
        SELECT image, invoice_number, issue_date, total_amount, status
        FROM invoices
        WHERE vendor_name_key = ?
          AND image != COALESCE(?, '')
          AND status != 'unusable'
        ORDER BY issue_date DESC
        LIMIT ?
        """,
        (key, current_image, limit),
    ).fetchall()

    if not rows:
        return {"found": 0, "invoices": [],
                "note": "No other invoices on file for this supplier."}

    return {
        "found": len(rows),
        "invoices": [
            {
                "invoice_number": r["invoice_number"],
                "issue_date": r["issue_date"],
                "total_amount": r["total_amount"],
                "status": r["status"],
            }
            for r in rows
        ],
        "note": "Excludes the invoice under review.",
    }


def find_similar_invoices(conn: sqlite3.Connection, vendor_name: str,
                          current_image: str | None = None) -> dict:
    """
    Work described like this invoice's, in this supplier's own history.

    Note what the model does NOT pass: the text to search with. The loop
    supplies it, built from the invoice under review. Letting the model
    choose the query text would let it search for something other than the
    document in front of it, which is not a question anyone asked.

    The supplier is resolved fuzzily first, then the search runs inside
    that supplier's invoices only. Searching everything would work here --
    48 documents -- and stop working at 10,000, where enough invoices
    describe similar work that the top score always looks alarming.
    """
    import embed

    row = conn.execute(
        "SELECT line_items FROM invoices WHERE image = ?", (current_image,)
    ).fetchone() if current_image else None

    if row is None or not row["line_items"]:
        return {"searched": False,
                "note": "This invoice has no readable line items, so there "
                        "is nothing to compare by description."}

    items = json.loads(row["line_items"])
    text = " | ".join(i["description"] for i in items
                      if i.get("description"))
    if not text:
        return {"searched": False,
                "note": "No item descriptions on this invoice."}

    keys = embed.vendor_keys_for(vendor_name)
    matches = embed.similar_invoices(text, n=5, exclude=current_image,
                                     vendor_keys=keys)

    if not matches:
        return {"searched": True, "found": 0, "matches": [],
                "searched_for": text,
                "note": "No other invoices on file from this supplier."}

    return {
        "searched": True,
        "found": len(matches),
        "searched_for": text,
        "matches": matches,
        "note": "Scores compare the work described, not the amounts. "
                "Recurring work scores high legitimately -- check the "
                "dates and amounts for a pattern before concluding.",
    }


def find_similar_vendors(vendor_name: str) -> dict:
    """Supplier names close to this one, for deciding if a vendor is new."""
    import embed

    matches = embed.similar_vendors(vendor_name, n=5)
    if not matches:
        return {"found": 0, "matches": [],
                "note": "No supplier names on file."}
    return {
        "found": len(matches),
        "searched_for": vendor_name,
        "matches": matches,
        "note": "Above ~0.85 is very likely the same company; below 0.5 is "
                "a different one.",
    }


def approve(reason: str) -> dict:
    """Terminal. Returns the decision; storing it is the loop's job."""
    return {"decision": "approve", "reason": reason}


def escalate(reason: str) -> dict:
    """Terminal."""
    return {"decision": "escalate", "reason": reason}


def rescan(reason: str) -> dict:
    """Terminal."""
    return {"decision": "rescan", "reason": reason}


# --------------------------------------------------------------------------
# The bridge
# --------------------------------------------------------------------------
def run_tool(conn: sqlite3.Connection, name: str, args: dict,
             current_image: str | None = None) -> dict:
    """
    Claude asks for a tool by name with a dict of arguments. This runs it.

    Everything the model sends is untrusted input -- it may invent a tool
    name, omit a required argument, or pass the wrong type. So nothing here
    assumes the call is well formed, and a bad call returns an {"error": ...}
    dict rather than raising.

    That matters more than it looks. An error handed BACK to the model is
    something it can recover from -- it reads "unknown tool" and tries a
    real one. An exception kills the run.

    `current_image` is context from the loop, not from the model. Only
    check_duplicate uses it, to avoid matching the document against itself.
    """
    if not isinstance(args, dict):
        return {"error": f"arguments must be an object, got {type(args).__name__}"}

    try:
        if name == "search_vendor":
            if not args.get("vendor_name"):
                return {"error": "search_vendor needs a vendor_name"}
            return search_vendor(conn, args["vendor_name"], current_image)

        if name == "check_duplicate":
            if not args.get("vendor_name") or not args.get("invoice_number"):
                return {"error": "check_duplicate needs vendor_name and "
                                 "invoice_number"}
            return check_duplicate(conn, args["vendor_name"],
                                   args["invoice_number"], current_image)

        if name == "find_similar_invoices":
            if not args.get("vendor_name"):
                return {"error": "find_similar_invoices needs a vendor_name"}
            return find_similar_invoices(conn, args["vendor_name"],
                                         current_image)

        if name == "find_similar_vendors":
            if not args.get("vendor_name"):
                return {"error": "find_similar_vendors needs a vendor_name"}
            return find_similar_vendors(args["vendor_name"])

        if name == "get_vendor_history":
            if not args.get("vendor_name"):
                return {"error": "get_vendor_history needs a vendor_name"}
            limit = args.get("limit", 10)
            limit = limit if isinstance(limit, int) and 0 < limit <= 50 else 10
            return get_vendor_history(conn, args["vendor_name"], limit,
                                      current_image)

        if name in DECISION_TOOLS:
            reason = args.get("reason")
            if not reason or not str(reason).strip():
                return {"error": f"{name} needs a reason"}
            return {"approve": approve, "escalate": escalate,
                    "rescan": rescan}[name](str(reason).strip())

        return {"error": f"unknown tool '{name}'. Available: "
                         f"{', '.join(TOOL_NAMES)}"}

    except sqlite3.Error as e:
        # A broken query must not take the whole batch down with it.
        return {"error": f"database error: {e}"}


# --------------------------------------------------------------------------
# Test it by hand, before any agent touches it
# --------------------------------------------------------------------------
def main() -> None:
    import json

    conn = connect()

    # Pick a real supplier out of the database to test against, rather than
    # hardcoding a name that may not be there.
    row = conn.execute(
        "SELECT display_name FROM vendors ORDER BY invoice_count DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise SystemExit("no vendors in the database -- run src/database.py first")
    real_name = row["display_name"]

    # Find a real duplicate pair to test check_duplicate against: the same
    # invoice number appearing on two different images.
    dup = conn.execute(
        """
        SELECT vendor_name, invoice_number, MIN(image) AS first_image
        FROM invoices
        WHERE invoice_number IS NOT NULL AND status != 'unusable'
        GROUP BY vendor_name_key, invoice_number
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()

    cases = [
        ("search_vendor",      {"vendor_name": real_name}, None),
        ("search_vendor",      {"vendor_name": "شركة لا توجد ابدا"}, None),
        ("get_vendor_history", {"vendor_name": real_name, "limit": 3}, None),
        ("find_similar_vendors", {"vendor_name": "مؤسسة النور"}, None),
        ("approve",            {"reason": "checks passed, amount typical"}, None),
        ("escalate",           {"reason": "94,018 SAR vs 14,200 average"}, None),
        ("rescan",             {"reason": "VAT number unreadable"}, None),
        # the malformed calls -- these must return errors, not explode
        ("search_vendor",      {}, None),
        ("approve",            {"reason": "  "}, None),
        ("teleport",           {"to": "Riyadh"}, None),
    ]

    # A semantic search needs a real image to build its query from.
    with_items = conn.execute(
        """
        SELECT image, vendor_name FROM invoices
        WHERE line_items IS NOT NULL AND line_items != '[]'
          AND status != 'unusable'
        ORDER BY image LIMIT 1
        """
    ).fetchone()
    if with_items:
        cases.append(("find_similar_invoices",
                      {"vendor_name": with_items["vendor_name"]},
                      with_items["image"]))

    if dup:
        # the real duplicate: searched from its second copy, so it finds the first
        cases.insert(1, ("check_duplicate",
                         {"vendor_name": dup["vendor_name"],
                          "invoice_number": dup["invoice_number"]},
                         "THIS_IMAGE_DOES_NOT_EXIST.png"))
        # and the same invoice excluding itself -- proves nothing matches itself
        cases.insert(2, ("check_duplicate",
                         {"vendor_name": dup["vendor_name"],
                          "invoice_number": "INV-0000-0000"},
                         dup["first_image"]))

    for name, args, image in cases:
        out = run_tool(conn, name, args, current_image=image)
        bad = "error" in out
        print(f"\n{'ERR ' if bad else '    '}{name}({json.dumps(args, ensure_ascii=False)})")
        print("     " + json.dumps(out, ensure_ascii=False, indent=2)
              .replace("\n", "\n     "))

    conn.close()
    print(f"\n{len(SCHEMAS)} tools, tested without a single API call.")


if __name__ == "__main__":
    main()