"""
Behavioural test cases for the duplicate detection.

    python src/scenarios.py          # insert them
    python src/scenarios.py --two    # only two months of the contract
    python src/scenarios.py --clean  # remove them
    python src/scenarios.py --list   # show what is loaded

WHY THESE EXIST
---------------
The real 50 invoices contain no semantic duplicates -- they are 50
independent one-off jobs, so nothing in them exercises the feature we just
built. Two cases have to be constructed, and the second is the one that
matters:

  GROUP A   the same job billed twice under a different number, reworded.
            check_duplicate is blind to it. Must be caught.

  GROUP B   a monthly maintenance contract: identical wording, identical
            amount, one invoice a month. Similarity will be 1.000 and that
            is CORRECT -- it is the same service, legitimately recurring.
            Must NOT be treated as a duplicate.

Anything can spot two identical invoices. The question is whether the
system can tell a double billing from a subscription, and group B is the
only way to find out. A duplicate detector that flags every recurring
invoice gets switched off in its first week.

WHAT THIS IS AND IS NOT
-----------------------
These are injected records, not rendered documents. They test the AGENT's
reasoning about similarity, which is the part that has never been
exercised; extraction is already measured properly against ground truth on
the real 50.

So: a behavioural fixture, not an end-to-end test. Worth saying plainly
rather than letting a reader assume these went through a camera.

They are excluded from the accuracy evaluation automatically, because
evaluate.py scores only images that appear in ground_truth.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import config
import embed
from database import connect, rebuild_vendors
from normalize import normalize_arabic

PREFIX = "scn_"


def item(description: str, qty: float, unit_price: float) -> dict:
    return {
        "description": description,
        "quantity": qty,
        "unit_price": unit_price,
        "line_total": round(qty * unit_price * 1.15, 2),
    }


def invoice(image: str, vendor: str, vat: str, number: str, date: str,
            subtotal: float, items: list[dict]) -> dict:
    """One fake record, in the same shape batch.py writes."""
    vat_amount = round(subtotal * 0.15, 2)
    return {
        "image": image,
        "status": "ok",
        "error": None,
        "invoice": {
            "vendor_name": vendor,
            "vendor_name_key": normalize_arabic(vendor),
            "vendor_vat_number": vat,
            "buyer_name": "شركة نجد للصناعات",
            "buyer_name_key": normalize_arabic("شركة نجد للصناعات"),
            "buyer_vat_number": "390838637940263",
            "invoice_number": number,
            "issue_date": date,
            "subtotal": subtotal,
            "vat_rate": 0.15,
            "vat_amount": vat_amount,
            "total_amount": round(subtotal + vat_amount, 2),
            "currency": "SAR",
            "line_items": items,
            "confidence": 0.95,
            "unreadable_fields": [],
        },
        "issues": [],
        "raw_fields": {},
        "meta": {"model": "fixture", "prompt_version": "fixture"},
        "retried": False,
        "recovered": False,
        "first_status": None,
        "first_error": None,
    }


# --------------------------------------------------------------------------
# Group A: the same work billed twice, reworded
# --------------------------------------------------------------------------
AMAL = "شركة الأمل للأنظمة"
AMAL_VAT = "312045067890123"

GROUP_A = [
    invoice(
        f"{PREFIX}a1.png", AMAL, AMAL_VAT, "INV-2026-5501", "2026-05-04",
        45000.00,
        [item("تركيب شبكات داخلية", 200, 180.00),
         item("توريد كابلات شبكة", 50, 180.00)],
    ),
    # Two weeks later. Different number, different wording, same money, same
    # job. This is the one the system has to catch.
    invoice(
        f"{PREFIX}a2.png", AMAL, AMAL_VAT, "INV-2026-5589", "2026-05-18",
        45000.00,
        [item("أعمال تمديد شبكات داخلية", 200, 180.00),
         item("تزويد أسلاك الشبكة", 50, 180.00)],
    ),
]

# --------------------------------------------------------------------------
# Group B: a legitimate monthly contract
# --------------------------------------------------------------------------
SIYANA = "مؤسسة الصيانة المتحدة"
SIYANA_VAT = "345678901234563"

GROUP_B = [
    invoice(
        f"{PREFIX}b{n}.png", SIYANA, SIYANA_VAT,
        # 25,000 rather than a realistic 2,500: the smaller figure is under
        # AUTO_APPROVE_LIMIT, so policy settled it in code and the agent
        # never saw it. A fixture that does not reach the thing it is
        # testing passes without testing anything.
        f"INV-2026-70{n:02d}", date, 25000.00,
        [item("خدمة صيانة دورية", 1, 25000.00)],
    )
    for n, date in enumerate(
        ["2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"], start=1
    )
]

ALL = GROUP_A + GROUP_B

EXPECTED = {
    f"{PREFIX}a2.png": ("escalate",
                        "same work as scn_a1, reworded, 14 days apart, "
                        "identical amount, different invoice number"),
    f"{PREFIX}b4.png": ("approve",
                        "fourth of a monthly series at a constant amount "
                        "-- a contract, not a duplicate"),
}

# With only two instalments on file there is no established cadence, so the
# honest answer is to escalate. Two identical invoices a month apart are as
# consistent with "the first was never paid, here it is again" as with "this
# is the second month of a contract".
EXPECTED_TWO = {
    f"{PREFIX}b2.png": ("escalate",
                        "only the second identical invoice -- two points do "
                        "not establish a monthly pattern"),
}


# --------------------------------------------------------------------------
def load(two_only: bool = False) -> None:
    """
    Insert the fixtures.

    two_only loads just the first TWO months of the contract, which is a
    different question from the four-month case: two invoices a month apart
    at the same amount could be a contract's second instalment, or the same
    bill sent again because the first went unpaid. You can draw a line
    through any two points -- it takes a third to show a rhythm.

    Loading all four lets the agent reason from invoices that, in
    production, would not exist yet: asked about February it answered
    "filling the February slot" using March and April. Correct given the
    data, impossible in real time.
    """
    from database import save_result

    records = GROUP_A + GROUP_B[:2] if two_only else ALL

    conn = connect()
    for record in records:
        save_result(conn, record)
    rebuild_vendors(conn)
    conn.commit()

    # Embed them too, or find_similar_invoices has nothing to find.
    c = embed.client()
    col = c.get_or_create_collection(embed.INVOICE_COLLECTION,
                                     metadata={"hnsw:space": "cosine"})
    vendors = c.get_or_create_collection(embed.VENDOR_COLLECTION,
                                         metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    for r in records:
        inv = r["invoice"]
        ids.append(r["image"])
        texts.append(embed.invoice_text(inv))
        metas.append({
            "invoice_number": inv["invoice_number"],
            "issue_date": inv["issue_date"],
            "total_amount": float(inv["total_amount"]),
            "vendor_name": inv["vendor_name"],
            "vendor_name_key": inv["vendor_name_key"],
            "item_count": len(inv["line_items"]),
        })
    col.upsert(ids=ids, embeddings=embed.embed(texts),
               documents=texts, metadatas=metas)

    names = sorted({r["invoice"]["vendor_name"] for r in records})
    vendors.upsert(
        ids=names,
        embeddings=embed.embed(names),
        documents=names,
        metadatas=[{"vendor_name_key": normalize_arabic(n)} for n in names],
    )

    conn.close()
    print(f"inserted {len(records)} fixture invoices, {len(names)} vendors\n")

    expected = EXPECTED_TWO if two_only else EXPECTED
    for image, (decision, why) in expected.items():
        print(f"  {image:16} expected: {decision.upper():9} -- {why}")
    print("\nnow run:")
    for image in expected:
        print(f"  python src/agent.py {image}")


def clean() -> None:
    conn = connect()
    n = conn.execute(
        f"SELECT COUNT(*) FROM invoices WHERE image LIKE '{PREFIX}%'"
    ).fetchone()[0]
    conn.execute(f"DELETE FROM invoices WHERE image LIKE '{PREFIX}%'")
    conn.execute(f"DELETE FROM decisions WHERE image LIKE '{PREFIX}%'")
    rebuild_vendors(conn)
    conn.commit()
    conn.close()

    c = embed.client()
    col = c.get_or_create_collection(embed.INVOICE_COLLECTION,
                                     metadata={"hnsw:space": "cosine"})
    col.delete(ids=[r["image"] for r in ALL])
    vendors = c.get_or_create_collection(embed.VENDOR_COLLECTION,
                                         metadata={"hnsw:space": "cosine"})
    vendors.delete(ids=sorted({r["invoice"]["vendor_name"] for r in ALL}))

    print(f"removed {n} fixture invoices")


def show() -> None:
    conn = connect()
    rows = conn.execute(
        f"""
        SELECT image, vendor_name, invoice_number, issue_date, total_amount,
               line_items
        FROM invoices WHERE image LIKE '{PREFIX}%' ORDER BY image
        """
    ).fetchall()
    conn.close()

    if not rows:
        print("no fixtures loaded")
        return
    for r in rows:
        items = json.loads(r["line_items"] or "[]")
        desc = " | ".join(i["description"] for i in items
                          if i.get("description"))
        print(f"  {r['image']:16} {r['invoice_number']:16} "
              f"{r['issue_date']:12} {r['total_amount']:>10,.2f}  "
              f"{r['vendor_name']}")
        print(f"                   {desc}")


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    elif "--list" in sys.argv:
        show()
    else:
        load(two_only="--two" in sys.argv)