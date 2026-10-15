"""
Look at the database from the terminal.

    python src/query.py                      # overview
    python src/query.py vendors              # all suppliers
    python src/query.py flagged              # anything needing attention
    python src/query.py vendor <name>        # one supplier's history
    python src/query.py sql "SELECT ..."     # any query you like

The last one is the useful one -- it means you can poke at the data
without opening another program.
"""

from __future__ import annotations

import sqlite3
import sys

from database import connect


def show(rows: list[sqlite3.Row], max_width: int = 34) -> None:
    """Print rows as an aligned table."""
    if not rows:
        print("  (no rows)")
        return

    cols = rows[0].keys()
    widths = {
        c: min(max_width, max(len(str(c)), *(len(str(r[c])) for r in rows)))
        for c in cols
    }

    print("  " + "  ".join(str(c).ljust(widths[c]) for c in cols))
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = "" if r[c] is None else str(r[c])
            if len(v) > widths[c]:
                v = v[: widths[c] - 1] + "…"
            cells.append(v.ljust(widths[c]))
        print("  " + "  ".join(cells))
    print(f"\n  {len(rows)} row(s)")


def overview(conn: sqlite3.Connection) -> None:
    print("\nby status:")
    show(conn.execute("""
        SELECT status,
               COUNT(*)                       AS count,
               ROUND(SUM(total_amount), 2)    AS value,
               ROUND(AVG(confidence), 2)      AS avg_conf
        FROM invoices GROUP BY status ORDER BY count DESC
    """).fetchall())

    print("\nmost recent 10:")
    show(conn.execute("""
        SELECT image, status, vendor_name, total_amount, issue_date
        FROM invoices ORDER BY issue_date DESC LIMIT 10
    """).fetchall())


def main() -> None:
    conn = connect()
    args = sys.argv[1:]
    cmd = args[0] if args else "overview"

    if cmd == "overview":
        overview(conn)

    elif cmd == "vendors":
        show(conn.execute("""
            SELECT display_name, invoice_count,
                   ROUND(total_billed, 2) AS total_billed,
                   first_seen, last_seen
            FROM vendors ORDER BY total_billed DESC
        """).fetchall())

    elif cmd == "flagged":
        show(conn.execute("""
            SELECT image, status, error_count, vendor_name, total_amount, error
            FROM invoices
            WHERE status != 'ok'
            ORDER BY error_count DESC, image
        """).fetchall())

    elif cmd == "vendor" and len(args) > 1:
        term = f"%{args[1]}%"
        show(conn.execute("""
            SELECT image, invoice_number, issue_date, total_amount, status
            FROM invoices
            WHERE vendor_name LIKE ? OR vendor_name_key LIKE ?
            ORDER BY issue_date DESC
        """, (term, term)).fetchall())

    elif cmd == "rejected":
        # On a rejected record the raw extraction is the only data left,
        # and it is usually mostly correct. Show what was readable.
        import json as _json
        rows = conn.execute("""
            SELECT image, error, fields_read, raw_fields
            FROM invoices WHERE status = 'unusable' ORDER BY image
        """).fetchall()
        if not rows:
            print("  (none)")
        for r in rows:
            print(f"\n  {r['image']}  ({r['fields_read']} field(s) read)")
            print(f"      rejected: {r['error']}")
            raw = _json.loads(r["raw_fields"] or "{}")
            for k, v in raw.items():
                mark = "  " if v not in (None, "", []) else "x "
                print(f"      {mark}{k:20} {v}")

    elif cmd == "sql" and len(args) > 1:
        try:
            show(conn.execute(args[1]).fetchall())
        except sqlite3.Error as e:
            print(f"  SQL error: {e}")

    else:
        print(__doc__)

    conn.close()


if __name__ == "__main__":
    main()
