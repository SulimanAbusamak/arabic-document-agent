"""
Turn invoices into vectors so they can be compared by meaning.

    python src/embed.py            # load every invoice into Chroma
    python src/embed.py --reset    # wipe and rebuild
    python src/embed.py similar dev_000.png    # what looks like this one?
    python src/embed.py vendor "مؤسسة النور"   # which supplier is this?

Nothing here calls an API. The embedding model runs locally and is free;
the only cost is a one-off ~470 MB download the first time.

WHY THIS EXISTS
---------------
check_duplicate finds the same invoice number from the same supplier. It
cannot find this:

    INV-2026-4410   مؤسسة النور   "تركيب شبكات داخلية"   45,000 SAR
    INV-2026-7781   مؤسسة النور   "أعمال تمديد شبكات"    45,000 SAR

Different numbers, different wording, same job billed twice. Exact matching
is blind to it. Comparing MEANING is not.

WHAT GOES IN THE VECTOR, AND WHAT DOES NOT
------------------------------------------
Only words: the supplier and what was bought.

Numbers stay OUT of the embedded text and live in the metadata instead.
Embedding models are poor at numeric magnitude -- "45,000.00" and
"54,000.00" come out almost identical, because as text they are nearly the
same characters. Putting amounts in the vector would make the similarity
score mean less, not more.

So the division of labour is:

    the vector   ->  does this describe the same work?     (fuzzy)
    the metadata ->  same amount? how many days apart?     (exact)

The agent gets both and weighs them. Arithmetic belongs in code; judgement
about wording belongs in the vector.
"""

from __future__ import annotations

import json
import sys

import config

# Multilingual, handles Arabic, 384 dimensions, runs on a laptop CPU.
# A model trained only on English would put every Arabic invoice in roughly
# the same place and the scores would be meaningless.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CHROMA_DIR = config.DATA_DIR / "chroma"
BATCH_RESULTS = config.DATA_DIR / "batch_results.json"

INVOICE_COLLECTION = "invoices"
VENDOR_COLLECTION = "vendors"

_model = None
_client = None


def model():
    """
    Load the embedding model once, on first use.

    Importing sentence_transformers takes a few seconds and loading the
    weights takes a few more, so it happens lazily -- a script that only
    reads from Chroma never pays for it.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"loading {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def client():
    """
    Chroma, stored on disk at data/chroma.

    PersistentClient rather than the in-memory one: the vectors survive
    between runs, so you embed once and query as often as you like.
    """
    global _client
    if _client is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    """Text in, vectors out. One list of 384 floats per string."""
    return model().encode(texts, normalize_embeddings=True).tolist()


# --------------------------------------------------------------------------
# What text represents an invoice
# --------------------------------------------------------------------------
def invoice_text(inv: dict) -> str:
    """
    The words that identify what this invoice is FOR -- the work, nothing
    else.

    The supplier name is deliberately NOT here, though it was in the first
    version. It is the longest piece of text on an invoice, so it dominated
    the vector: searching with the supplier name included returned exactly
    the same five results whether or not the supplier filter was applied,
    because the score was mostly measuring "same supplier" -- which the
    metadata filter already answers exactly, and far better.

    Measuring the same thing twice, once badly, made the score worse.

    Same principle as keeping the amounts out: anything you can filter
    exactly does not belong in the fuzzy part.

        'تركيب شبكات داخلية | أجهزة حاسب محمول'
    """
    return " | ".join(
        str(item["description"])
        for item in (inv.get("line_items") or [])
        if item.get("description")
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(reset: bool = False) -> None:
    """Embed every usable invoice and store it, plus one entry per vendor."""
    if not BATCH_RESULTS.exists():
        raise SystemExit(f"{BATCH_RESULTS} not found -- run src/batch.py first")

    results = json.loads(BATCH_RESULTS.read_text(encoding="utf-8"))
    c = client()

    if reset:
        for name in (INVOICE_COLLECTION, VENDOR_COLLECTION):
            try:
                c.delete_collection(name)
                print(f"deleted collection {name}")
            except Exception:
                pass

    invoices = c.get_or_create_collection(
        INVOICE_COLLECTION, metadata={"hnsw:space": "cosine"})
    vendors = c.get_or_create_collection(
        VENDOR_COLLECTION, metadata={"hnsw:space": "cosine"})

    # ---- invoices --------------------------------------------------------
    ids, texts, metas = [], [], []
    for r in results:
        inv = r.get("invoice")
        if not inv:                     # rejected documents have no record
            continue
        text = invoice_text(inv)
        if not text:
            continue
        ids.append(r["image"])
        texts.append(text)
        metas.append({
            # Everything the agent needs to judge a match EXACTLY, kept out
            # of the vector on purpose.
            "invoice_number": inv.get("invoice_number") or "",
            "issue_date": inv.get("issue_date") or "",
            "total_amount": float(inv.get("total_amount") or 0),
            "vendor_name": inv.get("vendor_name") or "",
            # The filter key. Searching all invoices works at 50 documents
            # and breaks at 10,000: with enough history, SOMETHING always
            # describes similar work, so the top similarity score stops
            # meaning anything. Narrowing to one supplier first is what
            # keeps the number meaningful -- and duplicate payment is a
            # per-supplier problem anyway. You do not pay the same bill
            # twice to two different companies.
            "vendor_name_key": inv.get("vendor_name_key") or "",
            "item_count": len(inv.get("line_items") or []),
        })

    if ids:
        invoices.upsert(ids=ids, embeddings=embed(texts),
                        documents=texts, metadatas=metas)
    print(f"{len(ids)} invoices embedded")

    # ---- vendors ---------------------------------------------------------
    # One entry per distinct supplier NAME as printed. The point is to catch
    # the variants the Day 2 normaliser cannot: it handles ة/ه and أ/ا, but
    # 'مؤسسة النور للتوريدات' and 'مؤسسة النور' flatten to different keys,
    # so the same company looks like two.
    seen: dict[str, str] = {}
    for r in results:
        inv = r.get("invoice") or {}
        name = inv.get("vendor_name")
        if name and name not in seen:
            seen[name] = inv.get("vendor_name_key") or ""

    if seen:
        names = list(seen)
        vendors.upsert(
            ids=names,
            embeddings=embed(names),
            documents=names,
            metadatas=[{"vendor_name_key": seen[n]} for n in names],
        )
    print(f"{len(seen)} vendor names embedded")
    print(f"stored in {CHROMA_DIR}")


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------
def vendor_keys_for(name: str, threshold: float = 0.85) -> list[str]:
    """
    Every normalised key that plausibly means the same company.

    The exact key alone is not enough. 'مؤسسة النور للتوريدات' and
    'مؤسسة النور' are one company under two trading names, and the Day 2
    normaliser flattens them to DIFFERENT keys -- it fixes spelling
    variants (ة/ه, أ/ا), not length variants.

    So filtering an invoice search on the exact key would hide the very
    duplicate it is looking for. The supplier has to be resolved FIRST,
    then the search runs across everything that supplier has ever been
    called.

    The threshold is deliberately high, because the two errors are not
    equally bad:

      a miss        one duplicate goes undetected; a human catches it.
      a false merge two real companies' histories get pooled, so every
                    average the agent reasons with is wrong -- silently,
                    and for every future invoice from either of them.

    0.85 is a starting point to be measured, not a fact.
    """
    keys = set()
    for match in similar_vendors(name, n=5):
        if match["similarity"] >= threshold and match["vendor_name_key"]:
            keys.add(match["vendor_name_key"])
    return sorted(keys)


def similar_invoices(text: str, n: int = 5,
                     exclude: str | None = None,
                     vendor_keys: list[str] | None = None) -> list[dict]:
    """
    The n invoices whose described work is closest to `text`.

    `vendor_keys` restricts the search to one supplier -- all the names that
    supplier is known by, from vendor_keys_for() -- and normally you want
    it. Nearest-neighbour search ALWAYS returns something, and the more
    history you have the closer the nearest thing is: across 10,000
    invoices, 200 of them are network installation jobs from 40 suppliers,
    so every network job matches every other at 0.90+ and the score stops
    carrying information. Inside one supplier's own history, 0.90 means
    something.

    The principle: SIMILARITY IS THE LAST FILTER, NOT THE FIRST. Narrow with
    the cheap exact facts -- supplier, date range, amount band -- and use the
    vector only to rank what is left.

    `exclude` leaves one image out -- the document under review, which would
    otherwise match itself perfectly. Same rule as every other lookup tool:
    the question is what we know APART from this document.

    Chroma returns a distance; we hand back a similarity, because that is
    what a human and a model both think in. Smaller-is-closer inverts the
    intuition and invites misreading.
    """
    col = client().get_or_create_collection(
        INVOICE_COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        return []

    if not vendor_keys:
        where = None
    elif len(vendor_keys) == 1:
        where = {"vendor_name_key": vendor_keys[0]}
    else:
        where = {"vendor_name_key": {"$in": vendor_keys}}

    # Ask for one extra, in case the excluded document comes back.
    res = col.query(query_embeddings=embed([text]),
                    n_results=min(n + 1, col.count()),
                    where=where)
    if not res["ids"][0]:
        return []

    out = []
    for i, doc_id in enumerate(res["ids"][0]):
        if exclude and doc_id == exclude:
            continue
        meta = res["metadatas"][0][i]
        out.append({
            "image": doc_id,
            "description": res["documents"][0][i],
            "similarity": round(1 - res["distances"][0][i], 3),
            "invoice_number": meta.get("invoice_number"),
            "issue_date": meta.get("issue_date"),
            "total_amount": meta.get("total_amount"),
            "vendor_name": meta.get("vendor_name"),
        })
    return out[:n]


def similar_vendors(name: str, n: int = 3) -> list[dict]:
    """
    Supplier names that look like this one.

    Catches what exact matching and the normaliser both miss: a trading name
    written long on one invoice and short on the next.
    """
    col = client().get_or_create_collection(
        VENDOR_COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        return []

    res = col.query(query_embeddings=embed([name]),
                    n_results=min(n, col.count()))
    return [
        {
            "vendor_name": res["documents"][0][i],
            "vendor_name_key": res["metadatas"][0][i].get("vendor_name_key"),
            "similarity": round(1 - res["distances"][0][i], 3),
        }
        for i in range(len(res["ids"][0]))
    ]


# --------------------------------------------------------------------------
def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "load"

    if cmd == "load":
        load(reset="--reset" in sys.argv)

    elif cmd == "similar" and len(args) > 1:
        image = args[1]
        results = json.loads(BATCH_RESULTS.read_text(encoding="utf-8"))
        match = next((r for r in results if r["image"] == image), None)
        if not match or not match.get("invoice"):
            raise SystemExit(f"{image} not found, or it has no record")

        inv = match["invoice"]
        text = invoice_text(inv)
        name = inv.get("vendor_name") or ""
        keys = vendor_keys_for(name)

        print(f"\nsupplier: {name}")
        print(f"resolved to {len(keys)} key(s): {keys}")
        print(f"query: {text}")

        # Both searches, side by side, so the difference the filter makes is
        # visible rather than asserted.
        print(f"\nsame supplier only:")
        rows = similar_invoices(text, n=5, exclude=image, vendor_keys=keys)
        if not rows:
            print("  (no other invoices from this supplier)")
        for m in rows:
            print(f"  {m['similarity']:.3f}  {m['image']:20} "
                  f"{m['invoice_number']:16} {m['issue_date']:12} "
                  f"{m['total_amount']:>12,.2f}")
            print(f"         {m['description']}")

        print(f"\nall suppliers (for comparison):")
        for m in similar_invoices(text, n=5, exclude=image):
            print(f"  {m['similarity']:.3f}  {m['image']:20} "
                  f"{m['invoice_number']:16} {m['issue_date']:12} "
                  f"{m['total_amount']:>12,.2f}")
            print(f"         {m['description']}")

    elif cmd == "vendor" and len(args) > 1:
        print(f"\nquery: {args[1]}\n")
        for m in similar_vendors(args[1], n=5):
            print(f"  {m['similarity']:.3f}  {m['vendor_name']}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()