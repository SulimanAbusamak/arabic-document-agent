"""
The screen a reviewer actually uses.

    streamlit run src/app.py

Three pages:

    Process   drop an invoice in, watch the whole pipeline run on it
    Review    browse what has been decided, and why
    Rules     set a standing exception, which is how a human's decision
              stops being a one-off

Everything here is a thin layer over functions that already existed. The
pipeline, the agent and the rules do not know this file exists, and that is
deliberate: a UI that contains logic is a UI you cannot test.

A note on how Streamlit works, because it explains the session_state code
below: the whole script re-runs top to bottom on every click. Anything you
want to survive a click has to live in st.session_state, or it is rebuilt
from scratch -- which here would mean re-reading an invoice through the API
every time someone expands a section.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import config
import embed
import rules
from agent import run_agent
from database import connect
from pipeline import process

st.set_page_config(page_title="Arabic Invoice Agent", layout="wide")

DECISION_COLOUR = {
    "approve": "🟢",
    "escalate": "🟡",
    "rescan": "🔴",
    "return_to_supplier": "🔴",
}

STATUS_COLOUR = {"ok": "🟢", "flagged": "🟡", "unusable": "🔴"}


@st.cache_resource
def db():
    conn = connect(same_thread=False)
    rules.init(conn)
    return conn


# --------------------------------------------------------------------------
# Shared bits of display
# --------------------------------------------------------------------------
def show_fields(inv: dict) -> None:
    """The extracted record, as a reviewer would want to read it."""
    left, right = st.columns(2)
    with left:
        st.markdown("**Supplier**")
        st.write(inv.get("vendor_name") or "—")
        st.caption(f"VAT {inv.get('vendor_vat_number') or '—'}")
        st.markdown("**Buyer**")
        st.write(inv.get("buyer_name") or "—")
        st.caption(f"VAT {inv.get('buyer_vat_number') or '—'}")
    with right:
        st.markdown("**Invoice**")
        st.write(inv.get("invoice_number") or "—")
        st.caption(f"issued {inv.get('issue_date') or '—'}")
        st.markdown("**Amounts**")
        cur = inv.get("currency") or ""
        st.write(f"subtotal {inv.get('subtotal'):,.2f} {cur}"
                 if inv.get("subtotal") is not None else "—")
        rate = inv.get("vat_rate")
        st.write(f"VAT {rate:.0%} — {inv.get('vat_amount'):,.2f} {cur}"
                 if rate is not None and inv.get("vat_amount") is not None
                 else "—")
        st.write(f"**total {inv.get('total_amount'):,.2f} {cur}**"
                 if inv.get("total_amount") is not None else "—")

    items = inv.get("line_items") or []
    if items:
        st.markdown("**Line items**")
        st.dataframe(items, hide_index=True, use_container_width=True)


def show_issues(issues: list[dict]) -> None:
    if not issues:
        st.success("No validation findings.")
        return
    for i in issues:
        text = f"**{i['rule']}** — {i['detail']}"
        if i["severity"] == "error":
            st.error(text)
        else:
            st.warning(text)


def show_decision(d) -> None:
    """An AgentDecision, or a database row with the same field names."""
    decision = d["decision"] if isinstance(d, dict) else d.decision
    reason = d["reason"] if isinstance(d, dict) else d.reason
    source = d["source"] if isinstance(d, dict) else d.source
    trace = d["trace"] if isinstance(d, dict) else d.trace

    st.markdown(f"## {DECISION_COLOUR.get(decision, '')} {decision.upper()}")
    st.write(reason)

    if isinstance(trace, str):
        trace = json.loads(trace or "[]")
    caption = f"decided by {source}"
    if trace:
        caption += "  ·  " + " → ".join(trace)
    st.caption(caption)


# --------------------------------------------------------------------------
# Page 1: process a new invoice
# --------------------------------------------------------------------------
def page_process() -> None:
    st.title("Process an invoice")
    st.caption("Drop an image in and the whole pipeline runs on it: read, "
               "normalise, validate, retry if needed, then decide.")

    upload = st.file_uploader("Invoice image", type=["png", "jpg", "jpeg"])
    if upload is None:
        st.info("Waiting for an invoice.")
        return

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / upload.name
    path.write_bytes(upload.getbuffer())

    # Re-processing on every click would mean paying for the same API call
    # repeatedly, so the result is keyed on the filename and kept.
    key = f"result::{upload.name}"
    if key not in st.session_state:
        with st.spinner("Reading the document…"):
            result = process(path)
        st.session_state[key] = result
    result = st.session_state[key]

    left, right = st.columns([1, 1])

    with left:
        st.image(str(path), use_container_width=True)

    with right:
        mark = STATUS_COLOUR.get(result.status, "")
        st.markdown(f"### {mark} {result.status}")

        if result.retried:
            if result.recovered:
                st.info(f"The first read failed ({result.first_status}). "
                        f"A second opinion from "
                        f"{result.meta.get('model')} did better and was kept.")
            else:
                st.warning("Two models were tried and neither did better. "
                           "That says the document is the problem, not the "
                           "model.")

        if result.invoice is None:
            st.error(result.error or "Could not build a record.")
            st.markdown("**What was readable**")
            st.json(result.raw_fields)
        else:
            show_fields(result.invoice.model_dump(mode="json"))
            st.divider()
            show_issues([i.model_dump() for i in result.issues])

    st.divider()

    # The decision is a separate, paid step, so it is a button rather than
    # something that happens on every page load.
    dkey = f"decision::{upload.name}"
    if dkey not in st.session_state:
        if st.button("Get a decision", type="primary"):
            with st.spinner("Checking history and duplicates…"):
                st.session_state[dkey] = run_agent(result, db())
            st.rerun()
    else:
        d = st.session_state[dkey]
        show_decision(d)
        if d.source == "agent":
            st.caption(f"{d.turns} turns · {d.input_tokens:,} in / "
                       f"{d.output_tokens:,} out · ${d.cost:.4f}")

        if st.button("Save to the database"):
            from database import save_decision, save_result
            from batch import result_to_dict

            conn = db()
            save_result(conn, result_to_dict(result))
            save_decision(conn, d)
            conn.commit()

            if result.invoice:
                _embed_one(result)
            st.success("Stored. It will be found by future duplicate checks.")


def _embed_one(result) -> None:
    """Add one processed invoice to the vector store."""
    inv = result.invoice.model_dump(mode="json")
    text = embed.invoice_text(inv)
    if not text:
        return
    col = embed.client().get_or_create_collection(
        embed.INVOICE_COLLECTION, metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[result.image],
        embeddings=embed.embed([text]),
        documents=[text],
        metadatas=[{
            "invoice_number": inv.get("invoice_number") or "",
            "issue_date": inv.get("issue_date") or "",
            "total_amount": float(inv.get("total_amount") or 0),
            "vendor_name": inv.get("vendor_name") or "",
            "vendor_name_key": inv.get("vendor_name_key") or "",
            "item_count": len(inv.get("line_items") or []),
        }],
    )


# --------------------------------------------------------------------------
# Page 2: review what has been decided
# --------------------------------------------------------------------------
def page_review() -> None:
    st.title("Review")
    conn = db()

    latest = """
        SELECT d.* FROM decisions d
        JOIN (SELECT image, MAX(id) AS id FROM decisions GROUP BY image) m
          ON m.id = d.id
    """

    counts = conn.execute(f"""
        SELECT decision, COUNT(*) n FROM ({latest}) GROUP BY decision
    """).fetchall()
    if not counts:
        st.info("Nothing decided yet. Run src/review.py, or process an "
                "invoice on the first page.")
        return

    cols = st.columns(len(counts))
    for col, row in zip(cols, counts):
        col.metric(f"{DECISION_COLOUR.get(row['decision'], '')} "
                   f"{row['decision']}", row["n"])

    st.divider()

    choice = st.selectbox(
        "Decision",
        ["all"] + sorted({r["decision"] for r in counts}),
    )
    where = "" if choice == "all" else f"WHERE d.decision = '{choice}'"

    rows = conn.execute(f"""
        SELECT d.*, i.status, i.vendor_name, i.total_amount, i.issue_date,
               i.issues, i.retried, i.recovered, i.raw_fields, i.line_items,
               i.invoice_number, i.vendor_vat_number, i.buyer_name,
               i.buyer_vat_number, i.subtotal, i.vat_rate, i.vat_amount,
               i.currency
        FROM ({latest}) d
        JOIN invoices i ON i.image = d.image
        {where}
        ORDER BY d.id DESC
    """).fetchall()

    for r in rows:
        header = (f"{DECISION_COLOUR.get(r['decision'], '')} {r['image']}  ·  "
                  f"{r['vendor_name'] or '—'}  ·  "
                  f"{(r['total_amount'] or 0):,.2f}")
        with st.expander(header):
            show_decision(dict(r))
            st.divider()
            inv = {k: r[k] for k in (
                "vendor_name", "vendor_vat_number", "buyer_name",
                "buyer_vat_number", "invoice_number", "issue_date",
                "subtotal", "vat_rate", "vat_amount", "total_amount",
                "currency")}
            inv["line_items"] = json.loads(r["line_items"] or "[]")
            show_fields(inv)
            show_issues(json.loads(r["issues"] or "[]"))

            # The point of the screen: turn this one decision into a rule.
            if r["decision"] == "escalate" and r["vendor_name"]:
                _rule_form(r)
                _override_form(r)


def _rule_form(row) -> None:
    """Set a standing rule straight from an escalated invoice."""
    st.divider()
    st.markdown("**This is a known, expected invoice**")
    st.caption("Recording why means the next one is decided without you. "
               "The rule is evidence the agent weighs, not a bypass — "
               "duplicates and validation still apply.")

    with st.form(f"rule::{row['image']}"):
        c1, c2, c3 = st.columns(3)
        kind = c1.selectbox("Rule", rules.RULE_TYPES,
                            key=f"k{row['image']}")
        amount = c2.number_input("Amount", value=float(row["total_amount"] or 0),
                                 key=f"a{row['image']}")
        cadence = c3.selectbox("Billed", rules.CADENCES,
                               key=f"c{row['image']}")
        note = st.text_input("Why — put the PO or contract number here",
                             key=f"n{row['image']}")
        c4, c5 = st.columns(2)
        who = c4.text_input("Set by", key=f"w{row['image']}")
        until = c5.date_input("Expires", key=f"u{row['image']}")

        if st.form_submit_button("Save rule"):
            if not note.strip() or not who.strip():
                st.error("A rule needs a reason and an owner.")
            else:
                rule_id = rules.add(
                    db(), row["vendor_name"], kind, amount, note, who,
                    expires_at=until.isoformat() if until else None,
                    cadence=cadence,
                )
                st.success(f"Rule #{rule_id} saved.")
def _override_form(row) -> None:
    """
    Let a reviewer return an invoice to the supplier.

    The agent is not allowed to make this call -- accusing a company of
    billing wrongly, on the strength of a model reading a scan, is not a
    decision software should take on its own. So the option exists only
    here, where a person has the document in front of them.

    It writes a NEW decision rather than editing the old one. The Review
    page shows the latest per invoice, so this one wins on screen, while
    the agent's original stays underneath. That is the audit trail: the
    system said escalate, a named person said return, on this date.
    """
    st.divider()
    st.markdown("**This invoice is wrong and should go back to the supplier**")
    st.caption("The agent never makes this call. Recording it here keeps "
               "both what the system decided and what you decided.")

    with st.form(f"override::{row['image']}"):
        why = st.text_area(
            "What is wrong with it — this goes in the draft to the supplier",
            key=f"ow{row['image']}")
        who = st.text_input("Returned by", key=f"op{row['image']}")

        if st.form_submit_button("Return to supplier"):
            if not why.strip() or not who.strip():
                st.error("A return needs a reason and a name.")
            else:
                from agent import AgentDecision
                from database import save_decision

                conn = db()
                d = AgentDecision(
                    image=row["image"],
                    decision="return_to_supplier",
                    reason=f"Returned by {who.strip()}: {why.strip()}",
                    source="human",
                    overridden=row["decision"],
                )
                save_decision(conn, d)
                conn.commit()
                st.success("Recorded. Draft below — send it yourself.")
                draft = _supplier_draft(row, why.strip(), who.strip())

                # Write it out rather than only showing it. A draft that
                # exists for one page render is a draft you lose by
                # pressing F5 -- and the decision is already saved by then,
                # so the record says a letter was prepared that nobody can
                # find.
                out = config.DATA_DIR / "escalations"
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{row['image']}.return.eml"
                path.write_text(draft, encoding="utf-8")

                st.success(f"Recorded. Draft saved as {path.name}.")
                st.code(draft)
                



def _supplier_draft(row, why: str, who: str) -> str:
    """
    A draft for the supplier. Deliberately a draft and not an email.

    Everything else the system writes goes to a colleague. This one goes
    outside the company, so a person reads it before it leaves.
    """
    total = row["total_amount"] or 0
    return f"""To: [supplier contact]
            Subject: Invoice {row['invoice_number'] or ''} — unable to process

            Dear Sir or Madam,

            We are unable to process the following invoice and are returning it for
            correction:

            Invoice number  {row['invoice_number'] or '—'}
            Date            {row['issue_date'] or '—'}
            Amount          {total:,.2f} {row['currency'] or 'SAR'}

            {why}

            Please issue a corrected invoice at your convenience. Payment will follow
            once we receive it.

            Kind regards,
            {who}
            """


# --------------------------------------------------------------------------
# Page 3: the rules
# --------------------------------------------------------------------------
def page_rules() -> None:
    st.title("Standing rules")
    st.caption("A human's decision, written down once, so the next matching "
               "invoice is decided without them.")

    conn = db()
    rows = conn.execute("SELECT * FROM vendor_rules ORDER BY id").fetchall()

    if not rows:
        st.info("No rules yet. Set one from an escalated invoice on the "
                "Review page.")
    today = date.today().isoformat()

    for r in rows:
        expired = r["expires_at"] and r["expires_at"] < today
        mark = "⚪" if expired else "🟢"
        with st.expander(f"{mark} #{r['id']}  {r['display_name']}  ·  "
                         f"{r['rule']}  {r['value']:,.2f}"):
            st.write(r["note"])
            cadence = r["cadence"] if "cadence" in r.keys() else None
            st.caption(
                f"billed {cadence or '—'} · set by {r['set_by']} on "
                f"{r['set_at'][:10]} · "
                + (f"expires {r['expires_at']}" if r["expires_at"]
                   else "no expiry")
                + ("  ·  EXPIRED" if expired else "")
            )
            if st.button("Remove", key=f"rm{r['id']}"):
                rules.remove(conn, r["id"])
                st.rerun()


# --------------------------------------------------------------------------
PAGES = {
    "Process": page_process,
    "Review": page_review,
    "Rules": page_rules,
}

choice = st.sidebar.radio("", list(PAGES))
st.sidebar.divider()
st.sidebar.caption(
    "Arabic invoice processing: Claude reads the document, validation checks "
    "the numbers against each other, and an agent checks them against "
    "history before deciding."
)
PAGES[choice]()
