"""
Standing exceptions a human has approved.

    python src/rules.py list
    python src/rules.py add <vendor> recurring <amount> --cadence monthly \
--note "..." --by "..." --until YYYY-MM-DD
    python src/rules.py add <vendor> limit <amount> --note "..." --by "..." \
--until YYYY-MM-DD
    python src/rules.py remove <id>

WHY THIS EXISTS
---------------
The agent escalates a monthly maintenance invoice because two instalments
do not yet prove a pattern. A human opens it, checks the contract, and
approves it. Next month the same invoice arrives and the agent escalates
it again, because nothing it learned was written down anywhere.

That is the gap this closes. A human resolves an escalation ONCE, records
why, and every future invoice matching that rule is decided without them.

This is also the honest answer to "how does the system improve in
production?" Not a better model, and no retraining: more rules, written by
the people who know which contracts exist. The knowledge lives in data the
decision logic reads.

TWO RULE TYPES
--------------
  recurring   approve invoices from this supplier AT THIS EXACT AMOUNT.
              For a fixed-fee contract: 28,750 every month, nothing else.

  limit       raise this supplier's auto-approve ceiling to this amount.
              For a supplier whose normal invoices are simply larger than
              the company-wide threshold.

WHAT A RULE CANNOT DO
---------------------
Override validation. A rule changes how much scrutiny an invoice needs; it
never makes a broken one payable. Unusable documents, failed arithmetic and
duplicates are all settled before any rule is consulted.

THREE THINGS THAT MAKE THIS SAFE RATHER THAN A HOLE IN THE CONTROLS
-------------------------------------------------------------------
  it expires     contracts end. A rule with no end date is a hole nobody
                 remembers opening.
  it is attributed  "approved by fatima@, PO-2026-114" is auditable.
                 "someone ticked a box in 2026" is not.
  it is narrow   'recurring' matches one amount, not one supplier. If they
                 bill ten times the usual next month it still escalates.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone

import config
from database import connect
from normalize import normalize_arabic

RULE_TYPES = ("recurring", "limit")

# How often the contract bills. Carried as a field rather than left in the
# note, because the agent has to reason about timing and anything it has to
# infer from prose it can infer wrongly.
#
# It did, on the first version: the note said "annual maintenance contract"
# and the agent concluded an annual contract should produce one invoice a
# YEAR, so it escalated the second monthly instalment. Sharp reading of an
# ambiguous sentence -- an annual contract billed monthly and one billed
# once a year are different things, and the note did not say which.
CADENCES = ("monthly", "quarterly", "annual", "none")


SCHEMA = """
CREATE TABLE IF NOT EXISTS vendor_rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_name_key TEXT    NOT NULL,       -- normalised, so spelling varies
    display_name    TEXT    NOT NULL,       -- as a human typed it
    rule            TEXT    NOT NULL,       -- recurring | limit
    value           REAL    NOT NULL,       -- the amount, or the ceiling
    cadence         TEXT,                   -- monthly | quarterly | annual | none
    note            TEXT    NOT NULL,       -- why. a PO number belongs here
    set_by          TEXT    NOT NULL,       -- who
    set_at          TEXT    NOT NULL,
    expires_at      TEXT                    -- NULL means no end date
);

CREATE INDEX IF NOT EXISTS idx_rules_vendor ON vendor_rules(vendor_name_key);
"""


def init(conn) -> None:
    """
    Create the table, and add the cadence column to one that predates it.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists,
    so a schema change needs its own step. ALTER TABLE raises if the column
    is already there, which is the normal case, so the error is swallowed.
    Crude as migrations go, but it means an existing database keeps its
    rules instead of needing a wipe.
    """
    conn.executescript(SCHEMA)
    try:
        conn.execute("ALTER TABLE vendor_rules ADD COLUMN cadence TEXT")
    except Exception:
        pass
    conn.commit()


# --------------------------------------------------------------------------
# Reading -- what policy_check uses
# --------------------------------------------------------------------------
def applicable(conn, vendor_name_key: str | None, total: float,
               today: str | None = None) -> dict | None:
    """
    The rule that lets this invoice through, if there is one.

    Returns None when no rule applies, which is the normal case -- the
    caller then falls back to the company-wide thresholds.

    Expired rules are ignored rather than deleted, so the record of what was
    approved and when survives.
    """
    if not vendor_name_key:
        return None
    today = today or date.today().isoformat()

    rows = conn.execute(
        """
        SELECT * FROM vendor_rules
        WHERE vendor_name_key = ?
          AND (expires_at IS NULL OR expires_at >= ?)
        ORDER BY id DESC
        """,
        (vendor_name_key, today),
    ).fetchall()

    for r in rows:
        if r["rule"] == "recurring":
            # Narrow on purpose: this exact amount, not this supplier.
            if abs(total - r["value"]) <= config.MONEY_TOLERANCE:
                return dict(r)
        elif r["rule"] == "limit":
            if total <= r["value"]:
                return dict(r)
    return None


def describe(rule: dict) -> str:
    """One line a reviewer can read in a decision reason."""
    if rule["rule"] == "recurring":
        kind = "recurring contract at {:,.2f} SAR".format(rule["value"])
        cadence = rule.get("cadence")
        if cadence and cadence != "none":
            kind += f", billed {cadence}"
    else:
        kind = "approval limit raised to {:,.2f} SAR".format(rule["value"])
    until = f", expires {rule['expires_at']}" if rule["expires_at"] else ""
    return (f"standing rule #{rule['id']}: {kind} for "
            f"{rule['display_name']} -- {rule['note']} "
            f"(set by {rule['set_by']} on {rule['set_at'][:10]}{until})")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def add(conn, vendor: str, rule: str, value: float, note: str,
        set_by: str, expires_at: str | None,
        cadence: str | None = None) -> int:
    if rule not in RULE_TYPES:
        raise SystemExit(f"rule must be one of {', '.join(RULE_TYPES)}")
    if not note.strip():
        raise SystemExit("a rule needs a note saying why it exists")
    if not set_by.strip():
        raise SystemExit("a rule needs to say who set it")
    if rule == "recurring" and not cadence:
        raise SystemExit("a recurring rule needs --cadence "
                         f"({'|'.join(CADENCES)}): how often it bills is "
                         "what decides whether a second invoice is expected")
    if cadence and cadence not in CADENCES:
        raise SystemExit(f"cadence must be one of {', '.join(CADENCES)}")

    cur = conn.execute(
        """
        INSERT INTO vendor_rules (vendor_name_key, display_name, rule, value,
                                  cadence, note, set_by, set_at, expires_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (normalize_arabic(vendor), vendor, rule, float(value), cadence,
         note.strip(), set_by.strip(),
         datetime.now(timezone.utc).isoformat(timespec="seconds"),
         expires_at),
    )
    conn.commit()
    return cur.lastrowid


def remove(conn, rule_id: int) -> bool:
    cur = conn.execute("DELETE FROM vendor_rules WHERE id = ?", (rule_id,))
    conn.commit()
    return cur.rowcount > 0


def show(conn) -> None:
    rows = conn.execute(
        "SELECT * FROM vendor_rules ORDER BY id"
    ).fetchall()
    if not rows:
        print("  no standing rules")
        return

    today = date.today().isoformat()
    for r in rows:
        expired = r["expires_at"] and r["expires_at"] < today
        mark = "EXPIRED" if expired else "active "
        cadence = r["cadence"] if "cadence" in r.keys() else None
        print(f"\n  #{r['id']}  {mark}  {r['rule']:9} "
              f"{r['value']:>12,.2f}  {cadence or '-':10} {r['display_name']}")
        print(f"          {r['note']}")
        print(f"          set by {r['set_by']} on {r['set_at'][:10]}"
              + (f", expires {r['expires_at']}" if r["expires_at"] else
                 ", no expiry"))
    print(f"\n  {len(rows)} rule(s)")


# --------------------------------------------------------------------------
def main() -> None:
    conn = connect()
    init(conn)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "list"

    def flag(name: str, default: str | None = None) -> str | None:
        key = f"--{name}"
        if key in sys.argv:
            i = sys.argv.index(key)
            if i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return default

    if cmd == "list":
        show(conn)

    elif cmd == "add" and len(args) >= 4:
        _, vendor, rule, value = args[:4]
        rule_id = add(
            conn, vendor, rule, float(value),
            note=flag("note", ""),
            set_by=flag("by", ""),
            expires_at=flag("until"),
            cadence=flag("cadence"),
        )
        print(f"added rule #{rule_id}")
        show(conn)

    elif cmd == "remove" and len(args) > 1:
        print("removed" if remove(conn, int(args[1])) else "no such rule")

    else:
        print(__doc__)

    conn.close()


if __name__ == "__main__":
    main()