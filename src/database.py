"""
Where processed invoices live.

    python src/database.py            # create the db and load batch results
    python src/database.py --reset    # wipe and rebuild from scratch

Plain sqlite3 from the standard library -- no ORM, no server, no setup.
A single-writer document pipeline does not need Postgres; the schema and
queries here are standard SQL, so moving later is a connection change.

Two tables:

  invoices   one row per processed document
  vendors    one row per distinct supplier, built from the invoices

The vendors table is what makes "have we dealt with this company before?"
answerable -- which is the lookup the agent needs tomorrow. It is keyed on
the NORMALISED name, so مؤسسة النور and مؤسسه النور land on the same row.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import config

DB_PATH = config.DATA_DIR / "invoices.db"
BATCH_RESULTS = config.DATA_DIR / "batch_results.json"


SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    image             TEXT    NOT NULL UNIQUE,
    status            TEXT    NOT NULL,          -- ok | flagged | unusable
    error             TEXT,

    vendor_name       TEXT,
    vendor_name_key   TEXT,                      -- normalised, for matching
    vendor_vat_number TEXT,
    buyer_name        TEXT,
    buyer_vat_number  TEXT,

    invoice_number    TEXT,
    issue_date        TEXT,                      -- ISO 'YYYY-MM-DD'
    subtotal          REAL,
    vat_rate          REAL,
    vat_amount        REAL,
    total_amount      REAL,
    currency          TEXT,

    confidence        REAL,
    issues            TEXT,                      -- JSON list
    issue_count       INTEGER DEFAULT 0,
    error_count       INTEGER DEFAULT 0,         -- issues of severity 'error'

    -- What the model actually returned, before validation rejected anything.
    -- On an 'unusable' row this is the ONLY surviving data, and it is often
    -- mostly correct: one missing required field rejects the record, but the
    -- other six fields may have read fine. A human reviewing the document
    -- needs to see them.
    raw_fields        TEXT,                      -- JSON
    fields_read       INTEGER DEFAULT 0,         -- how many came back non-null

    model             TEXT,
    prompt_version    TEXT,
    processed_at      TEXT NOT NULL
);

-- Indexes: columns the agent will filter and sort on.
CREATE INDEX IF NOT EXISTS idx_vendor_key ON invoices(vendor_name_key);
CREATE INDEX IF NOT EXISTS idx_status     ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_issue_date ON invoices(issue_date);


CREATE TABLE IF NOT EXISTS decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    image          TEXT    NOT NULL,
    decision       TEXT    NOT NULL,          -- approve | escalate | reject
    reason         TEXT    NOT NULL,
    source         TEXT    NOT NULL,          -- policy | agent
    overridden     TEXT,                      -- set if policy vetoed the model
    trace          TEXT,                      -- JSON list of tools called
    turns          INTEGER DEFAULT 0,
    input_tokens   INTEGER DEFAULT 0,
    output_tokens  INTEGER DEFAULT 0,
    model          TEXT,
    decided_at     TEXT    NOT NULL
);

-- Deliberately NOT unique on image. The invoices table is unique per
-- document because a document is a fact; a decision is a judgement, and one
-- document can be judged more than once -- re-run after a prompt change, or
-- escalated by the agent and later approved by a human. A column could only
-- hold the latest; a table keeps the history.
CREATE INDEX IF NOT EXISTS idx_dec_image    ON decisions(image);
CREATE INDEX IF NOT EXISTS idx_dec_decision ON decisions(decision);

CREATE TABLE IF NOT EXISTS vendors (
    vendor_name_key   TEXT PRIMARY KEY,
    display_name      TEXT NOT NULL,             -- real spelling, for humans
    vat_number        TEXT,
    invoice_count     INTEGER NOT NULL DEFAULT 0,
    total_billed      REAL    NOT NULL DEFAULT 0,
    first_seen        TEXT,
    last_seen         TEXT
);
"""


def connect() -> sqlite3.Connection:
    """
    Open the database.

    row_factory = sqlite3.Row makes rows behave like dicts, so you write
    row["vendor_name"] instead of row[4] and nothing breaks when a column
    is added in the middle.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(reset: bool = False) -> None:
    """Create the tables. With reset=True, delete the file first."""
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"deleted {DB_PATH.name}")
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def save_result(conn: sqlite3.Connection, result: dict) -> None:
    """
    Store one processed document.

    'INSERT OR REPLACE' + the UNIQUE constraint on image means re-running
    updates the existing row instead of creating duplicates. Without it,
    running the loader twice would double every record.
    """
    inv = result.get("invoice") or {}
    issues = result.get("issues") or []
    meta = result.get("meta") or {}
    raw = result.get("raw_fields") or {}

    # How much did the model actually manage to read? On a rejected record
    # this is the difference between "nothing legible" and "one field short".
    fields_read = sum(
        1 for k, v in raw.items()
        if k != "unreadable_fields" and v not in (None, "", [])
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO invoices (
            image, status, error,
            vendor_name, vendor_name_key, vendor_vat_number,
            buyer_name, buyer_vat_number,
            invoice_number, issue_date,
            subtotal, vat_rate, vat_amount, total_amount, currency,
            confidence, issues, issue_count, error_count,
            raw_fields, fields_read,
            model, prompt_version, processed_at
        ) VALUES (?,?,?, ?,?,?, ?,?, ?,?, ?,?,?,?,?, ?,?,?,?, ?,?, ?,?,?)
        """,
        (
            result["image"], result["status"], result.get("error"),
            inv.get("vendor_name"), inv.get("vendor_name_key"),
            inv.get("vendor_vat_number"),
            inv.get("buyer_name"), inv.get("buyer_vat_number"),
            inv.get("invoice_number"), inv.get("issue_date"),
            inv.get("subtotal"), inv.get("vat_rate"),
            inv.get("vat_amount"), inv.get("total_amount"), inv.get("currency"),
            # confidence comes from the model, so it survives rejection
            inv.get("confidence", raw.get("confidence")),
            json.dumps(issues, ensure_ascii=False),
            len(issues),
            sum(1 for i in issues if i.get("severity") == "error"),
            json.dumps(raw, ensure_ascii=False),
            fields_read,
            meta.get("model"), meta.get("prompt_version"),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def rebuild_vendors(conn: sqlite3.Connection) -> int:
    """
    Recompute the vendors table from the invoices.

    Derived, not maintained by hand -- so it can never drift out of sync
    with the invoices it summarises. Only usable records count toward the
    totals; a rejected document should not create a vendor.
    """
    conn.execute("DELETE FROM vendors")
    conn.execute(
        """
        INSERT INTO vendors (vendor_name_key, display_name, vat_number,
                             invoice_count, total_billed, first_seen, last_seen)
        SELECT
            vendor_name_key,
            MIN(vendor_name),                 -- one consistent spelling
            MAX(vendor_vat_number),           -- whichever VAT number we have
            COUNT(*),
            COALESCE(SUM(total_amount), 0),
            MIN(issue_date),
            MAX(issue_date)
        FROM invoices
        WHERE vendor_name_key IS NOT NULL
          AND status != 'unusable'
        GROUP BY vendor_name_key
        """
    )
    return conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]

def save_decision(conn: sqlite3.Connection, d) -> None:
    """Record one agent or policy decision. Appends; never overwrites."""
    conn.execute(
        """
        INSERT INTO decisions (
            image, decision, reason, source, overridden,
            trace, turns, input_tokens, output_tokens, model, decided_at
        ) VALUES (?,?,?,?,?, ?,?,?,?,?,?)
        """,
        (
            d.image, d.decision, d.reason, d.source, d.overridden,
            json.dumps(d.trace, ensure_ascii=False),
            d.turns, d.input_tokens, d.output_tokens,
            config.MODEL,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def decided_images(conn: sqlite3.Connection) -> set[str]:
    """Which documents already have a decision, so a re-run can skip them."""
    return {r["image"] for r in
            conn.execute("SELECT DISTINCT image FROM decisions")}

def load_batch_results(reset: bool = False) -> None:
    """Read batch_results.json into the database."""
    if not BATCH_RESULTS.exists():
        raise SystemExit(f"{BATCH_RESULTS} not found -- run src/batch.py first")

    init(reset=reset)
    results = json.loads(BATCH_RESULTS.read_text(encoding="utf-8"))

    conn = connect()
    for result in results:
        save_result(conn, result)
    n_vendors = rebuild_vendors(conn)
    conn.commit()

    n_invoices = conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
    conn.close()

    print(f"loaded {n_invoices} invoices, {n_vendors} distinct vendors")


# --------------------------------------------------------------------------
# Reading -- what the agent will use tomorrow
# --------------------------------------------------------------------------
def find_vendor(conn: sqlite3.Connection, name_key: str) -> sqlite3.Row | None:
    """
    Have we dealt with this supplier before?

    Looks up by NORMALISED key, so a different spelling of the same name
    still finds the record.
    """
    return conn.execute(
        "SELECT * FROM vendors WHERE vendor_name_key = ?", (name_key,)
    ).fetchone()


def vendor_history(conn: sqlite3.Connection, name_key: str,
                   limit: int = 10) -> list[sqlite3.Row]:
    """Recent invoices from one supplier, newest first."""
    return conn.execute(
        """
        SELECT image, invoice_number, issue_date, total_amount, status
        FROM invoices
        WHERE vendor_name_key = ? AND status != 'unusable'
        ORDER BY issue_date DESC
        LIMIT ?
        """,
        (name_key, limit),
    ).fetchall()


def summary(conn: sqlite3.Connection) -> dict:
    """Headline numbers, for the terminal and later for a dashboard."""
    row = conn.execute(
        """
        SELECT COUNT(*)                                        AS total,
               SUM(status = 'ok')                              AS ok,
               SUM(status = 'flagged')                         AS flagged,
               SUM(status = 'unusable')                        AS unusable,
               SUM(CASE WHEN status != 'unusable'
                        THEN total_amount ELSE 0 END)          AS value,
               AVG(CASE WHEN status != 'unusable'
                        THEN confidence END)                   AS avg_confidence
        FROM invoices
        """
    ).fetchone()
    return dict(row)


# --------------------------------------------------------------------------
def main() -> None:
    reset = "--reset" in sys.argv
    load_batch_results(reset=reset)

    conn = connect()
    s = summary(conn)

    print(f"\n{'-' * 46}")
    print(f"  invoices    {s['total']}")
    print(f"  ok          {s['ok']}")
    print(f"  flagged     {s['flagged']}")
    print(f"  unusable    {s['unusable']}")
    print(f"  total value {s['value']:,.2f} SAR")
    print(f"  avg conf    {s['avg_confidence']:.2f}")
    print(f"{'-' * 46}")

    print("\ntop vendors by value:")
    rows = conn.execute(
        """
        SELECT display_name, invoice_count, total_billed
        FROM vendors ORDER BY total_billed DESC LIMIT 5
        """
    ).fetchall()
    for r in rows:
        print(f"  {r['invoice_count']:>2} inv  {r['total_billed']:>12,.2f}  "
              f"{r['display_name']}")

    # Prove the matching works: look one up by its normalised key.
    first = conn.execute(
        "SELECT vendor_name_key, display_name FROM vendors LIMIT 1"
    ).fetchone()
    if first:
        v = find_vendor(conn, first["vendor_name_key"])
        print(f"\nlookup '{first['display_name']}' -> "
              f"{v['invoice_count']} invoice(s), "
              f"{v['total_billed']:,.2f} SAR total")

    conn.close()
    print(f"\ndatabase: {DB_PATH}")


if __name__ == "__main__":
    main()
