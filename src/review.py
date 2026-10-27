"""
Run the agent over every invoice and record what it decided.

    python src/review.py              # everything not yet decided
    python src/review.py dev          # dev only
    python src/review.py --force      # re-decide everything (costs money)

Already-decided documents are skipped by default, so re-running after a
crash or an interruption costs nothing. Use --force only when you have
changed the prompt or the tools and want fresh decisions -- the old ones
stay in the table either way, so you can compare.

Each decision is committed as it happens rather than all at once at the
end. A run that dies on invoice 40 keeps the 39 decisions you already paid
for.
"""

from __future__ import annotations

import sys
import time

import config
import rules
from agent import AgentDecision, run_agent
from batch import collect_images
from database import connect, decided_images, save_decision
from pipeline import process


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0] if args else "all"
    force = "--force" in sys.argv

    images = collect_images(which)
    if not images:
        raise SystemExit(f"no images found for '{which}'")

    conn = connect()
    rules.init(conn)           
    already = set() if force else decided_images(conn)
    todo = [p for p in images if p.name not in already]

    print(f"{len(images)} invoices, {len(already)} already decided, "
          f"{len(todo)} to do\n")
    if not todo:
        print("nothing to do -- use --force to re-decide")
        show_summary(conn)
        conn.close()
        return

    decisions: list[AgentDecision] = []
    started = time.time()

    for n, path in enumerate(todo, start=1):
        result = process(path)                 # cached, so free
        d = run_agent(result, conn)
        save_decision(conn, d)
        conn.commit()                          # keep what we paid for
        decisions.append(d)

        where = "policy" if d.source == "policy" else f"agent/{d.turns}"
        note = "  <- recovered by retry" if result.recovered else ""
        print(f"  [{n:>2}/{len(todo)}] {d.decision.upper():18} "
              f"{where:9} {path.name}{note}")

    elapsed = time.time() - started

    # ---- what it cost ----------------------------------------------------
    by_agent = [d for d in decisions if d.source == "agent"]
    tokens_in = sum(d.input_tokens for d in decisions)
    tokens_out = sum(d.output_tokens for d in decisions)
    cost = sum(d.cost for d in decisions)

    print(f"\n{'-' * 52}")
    print(f"  decided      {len(decisions)}")
    print(f"  by policy    {len(decisions) - len(by_agent)}  "
          f"(free -- no API call)")
    print(f"  by agent     {len(by_agent)}")
    print(f"  tokens       in {tokens_in:,}  out {tokens_out:,}")
    print(f"  cost         ${cost:.4f}")
    print(f"  time         {elapsed:.1f}s")

    show_summary(conn)
    conn.close()


def show_summary(conn) -> None:
    """
    Read the breakdown back out of the table, not out of memory.

    Only the LATEST decision per invoice counts. The table keeps every
    decision ever made -- that history is the point of it -- but a summary
    that adds two runs together tells you nothing, which is exactly the bug
    this had the first time.
    """
    latest = """
        SELECT d.* FROM decisions d
        JOIN (SELECT image, MAX(id) AS id FROM decisions GROUP BY image) m
          ON m.id = d.id
    """

    print(f"\n{'-' * 52}")
    runs = conn.execute("SELECT COUNT(*) n FROM decisions").fetchone()["n"]
    kept = conn.execute(f"SELECT COUNT(*) n FROM ({latest})").fetchone()["n"]
    print(f"  {kept} invoices, {runs} decisions on record "
          f"(showing the latest for each)")

    print("\n  decisions:")
    for r in conn.execute(f"""
        SELECT decision, source, COUNT(*) AS n
        FROM ({latest}) GROUP BY decision, source
        ORDER BY decision, source
    """):
        print(f"    {r['decision']:18} {r['source']:7} {r['n']:>3}")

    print("\n    approve             pay it")
    print("    escalate            a person here must look at it")
    print("    rescan              we cannot read it -- fetch a better copy")
    print("    return_to_supplier  set by a human after review, never by "
          "the system")

    # The interesting cross-tab: what the pipeline thought, against what was
    # decided. A 'flagged' invoice that got approved is worth a hard look.
    print("\n  pipeline status vs decision:")
    for r in conn.execute(f"""
        SELECT i.status, d.decision, COUNT(*) AS n
        FROM ({latest}) d JOIN invoices i ON i.image = d.image
        GROUP BY i.status, d.decision
        ORDER BY i.status, d.decision
    """):
        print(f"    {r['status']:9} -> {r['decision']:18} {r['n']:>3}")

    # Did the retry change any outcomes? A document recovered from flagged to
    # ok should now be approvable, and that is the whole payoff of the
    # cascade expressed as decisions rather than as accuracy.
    rows = conn.execute(f"""
        SELECT i.retried, i.recovered, d.decision, COUNT(*) AS n
        FROM ({latest}) d JOIN invoices i ON i.image = d.image
        WHERE i.retried = 1
        GROUP BY i.retried, i.recovered, d.decision
        ORDER BY i.recovered DESC, d.decision
    """).fetchall()
    if rows:
        print("\n  documents that needed a second opinion:")
        for r in rows:
            tag = "recovered" if r["recovered"] else "not recovered"
            print(f"    {tag:15} -> {r['decision']:18} {r['n']:>3}")

    row = conn.execute(f"""
        SELECT COUNT(*) AS n, SUM(input_tokens) AS ti, SUM(output_tokens) AS t_o
        FROM ({latest}) WHERE source = 'agent'
    """).fetchone()
    if row["n"]:
        total = config.call_cost(config.MODEL, row["ti"] or 0, row["t_o"] or 0)
        print(f"\n  agent runs   {row['n']}   total ${total:.4f}   "
              f"(${total / row['n']:.4f} each)")

    overrides = conn.execute(
        f"SELECT image FROM ({latest}) WHERE overridden IS NOT NULL"
    ).fetchall()
    if overrides:
        print(f"\n  policy overrode the agent on {len(overrides)}:")
        for r in overrides:
            print(f"    {r['image']}")


if __name__ == "__main__":
    main()
