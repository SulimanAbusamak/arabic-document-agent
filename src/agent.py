"""
The agent: decide what to do with one validated invoice.

    python src/agent.py dev_000          # one invoice
    python src/agent.py dev_012_deg      # the duplicate
    python src/agent.py dev_022          # the one with the misread digits

Day 2 asked Claude a question and got an answer. This is different: Claude
is handed a goal and six tools, and it chooses which to call, in what
order, and what to conclude. We write the tools; the model does the
reasoning.

Two things to understand before reading the code.

1. THE MODEL DOES NOT DECIDE EVERYTHING.
   `policy_check()` runs FIRST, in plain Python, and settles the cases that
   are policy rather than judgement -- unusable documents, failed
   validation, anything over 50,000 SAR. Those never reach the model. An
   LLM is good at "this amount looks strange for this supplier" and bad at
   being a hard limit: ask it whether a two-million-riyal invoice needs a
   human and it will *usually* say yes, and "usually" is not a control you
   can put in front of a finance team.

   So: hard limits in code, judgement in the model, and the model only gets
   the middle band.

2. NOTHING IS WRITTEN HERE.
   `run_agent()` returns an AgentDecision and touches no table. Storing is
   review.py's job. Same reason the write tools don't write: it means you
   can run this file on the same invoice twenty times while tuning the
   prompt without corrupting anything.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Literal

import anthropic

import config
import tools
from database import connect
from pipeline import Result, find_image, process

Decision = Literal["approve", "escalate", "rescan", "return_to_supplier"]

# Four outcomes, not three, and the split is about WHO has to act:
#
#   approve             pay it.
#   escalate            a person here must look at it.
#   rescan              we cannot read the paper. OUR problem -- fetch a
#                       better copy. No contact with the supplier.
#   return_to_supplier  the invoice itself is wrong. THEIR problem -- ask
#                       for a corrected one.
#
# Note what is NOT here: nothing in the system ever decides
# return_to_supplier on its own. It is set by a human after an escalation,
# and the reason is this project's own evaluation:
#
#   dev_022's arithmetic did not reconcile. The obvious reading is that the
#   supplier made a mistake. The ground truth says otherwise -- OUR read was
#   wrong, Haiku saw ١٢ as ١٣, and Sonnet later read it correctly.
#
# Had the system auto-returned on a failed arithmetic check, it would have
# written to a Saudi company accusing them of an error we had made. From
# inside the pipeline the two cases are indistinguishable: "subtotal + VAT
# does not equal total" means either their invoice is wrong or we misread a
# digit, and only someone holding the original can tell which.

# A run that has not decided after this many round trips is a run that is
# going in circles. Cap it: an unbounded while-loop against a paid API is
# how you wake up to a surprising bill.
MAX_TURNS = 6


SYSTEM = f"""You are an accounts-payable reviewer at a Saudi company. You are \
given one invoice that has already been read and validated by an automated \
pipeline, and you decide what happens to it.

You have tools to look up the supplier, their invoice history, and whether \
this invoice has already been processed. Use them before deciding. You \
cannot see the original document -- the extracted fields and the validation \
findings are all you have, so if something important is missing or \
unreadable, say so rather than guessing.

Your job is to reach exactly one decision: approve, escalate or rescan.

Never accuse the supplier of an error. If the numbers on an invoice do not \
reconcile, that is just as likely to be a digit misread on our side as a \
mistake on theirs, and you cannot see the original document to tell the \
difference. Escalate instead, and say what does not add up.

How to weigh things:

- A duplicate is never approved. Paying the same invoice twice is the most \
common and most expensive error in this job.
- An amount far out of line with what this supplier normally bills is worth \
escalating even when every validation check passed. Compare against their \
average before you judge.
- A supplier we have never processed before, billing a substantial amount, \
deserves a human's eyes.
- A validation warning is a reason to look harder, not an automatic refusal.
- Escalating costs a reviewer five minutes. A wrong approval costs money and \
is hard to undo. When genuinely unsure, escalate.

What NOT to treat as findings, because they are artefacts of how documents \
reach you rather than facts about this invoice:

- Documents are processed in arbitrary order, not chronologically. An invoice \
dated earlier than others already on file is completely normal. Date \
sequence relative to the supplier's history is not a finding.
- The supplier's history may itself contain invoices that are under review or \
already known to be duplicated. That is someone else's open item, not a \
reason to hold THIS invoice. Judge the document in front of you.
- Judge the supplier's identity on the VAT number of the invoice under \
review against the number search_vendor reports, and nothing else.

Invoices under {config.AUTO_APPROVE_LIMIT:,} SAR with no findings and no \
duplicate have already been approved before reaching you, and anything over \
{config.ESCALATION_LIMIT:,} SAR has already been escalated. So every invoice \
you see is a real judgement call.

Give your reason in one sentence, with the numbers you compared. A reviewer \
reads it without seeing any of your tool calls."""


@dataclass
class AgentDecision:
    """What came out of one run."""
    image: str
    decision: Decision
    reason: str
    source: Literal["policy", "agent"]      # who decided
    trace: list[str] = field(default_factory=list)   # tools called, in order
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    overridden: str | None = None           # set if policy vetoed the model

    @property
    def cost(self) -> float:
        return (self.input_tokens * 0.10 + self.output_tokens * 0.50) / 1e6


# --------------------------------------------------------------------------
# The hard rules -- code, not judgement
# --------------------------------------------------------------------------
def reference(raw: dict) -> str:
    """
    Identify a document from whatever was legible, for a human to find it.

    A rejected invoice is not a blank page. On dev_001_deg the supplier name
    was unreadable and SEVEN other fields read perfectly -- the number, the
    date, all three amounts. A reason that says only "validation failed"
    throws all of that away and leaves a clerk with nothing to search for.
    """
    bits = []
    if raw.get("invoice_number"):
        bits.append(f"invoice {raw['invoice_number']}")
    if raw.get("issue_date"):
        bits.append(f"dated {raw['issue_date']}")
    if raw.get("total_amount"):
        bits.append(f"total {raw['total_amount']}")
    if raw.get("vendor_name"):
        bits.append(f"from {raw['vendor_name']}")
    elif raw.get("buyer_name"):
        bits.append(f"addressed to {raw['buyer_name']}")
    return ", ".join(bits) if bits else "no identifying field was legible"


def policy_check(result: Result, conn) -> AgentDecision | None:
    """
    Settle the cases that are not the model's to decide.

    Returns a finished AgentDecision, or None meaning "this one needs
    judgement -- hand it to the agent". Every path here is free: no API
    call, so roughly a third of the batch never costs anything.
    """
    inv = result.invoice
    raw = result.raw_fields or {}

    # 1. The document could not be read. Ours to fix, not the supplier's.
    if result.status == "unusable" or inv is None:
        unreadable = raw.get("unreadable_fields") or []
        what = (", ".join(unreadable) if unreadable
                else (result.error or "unknown"))

        # Two independent models failing is a stronger claim than one: it
        # says the ink is not there, rather than that we used a weak model.
        confidence = (
            "Two models were tried and neither could read it, so a better "
            "copy is needed rather than a better model."
            if result.retried else
            "A second model was not attempted."
        )
        return AgentDecision(
            image=result.image, decision="rescan", source="policy",
            reason=(f"Could not be read: {what}. {reference(raw)}. "
                    f"{confidence} Request a clearer copy."),
        )

    # 2. Validation found a hard error: the numbers on the document do not
    #    agree with each other. Escalate -- and say plainly that this could
    #    be their mistake OR ours, because it could be either and the
    #    system has no way to tell.
    errors = [i for i in result.issues if i.severity == "error"]
    if errors:
        detail = "; ".join(i.detail for i in errors)
        tried = (" A second model was already tried and did not resolve it."
                 if result.retried else "")
        return AgentDecision(
            image=result.image, decision="escalate", source="policy",
            reason=(f"Amounts do not reconcile: {detail}. This is either an "
                    f"error on the invoice or a misread digit on our side -- "
                    f"compare against the original before contacting the "
                    f"supplier.{tried}"),
        )

    total = inv.total_amount or 0

    # 3. Above the escalation limit, a human signs off. Always. No amount
    #    of looking clean gets an invoice this size paid automatically.
    if total > config.ESCALATION_LIMIT:
        return AgentDecision(
            image=result.image, decision="escalate", source="policy",
            reason=(f"{total:,.2f} SAR exceeds the "
                    f"{config.ESCALATION_LIMIT:,} SAR approval limit."),
        )

    # 4. Small, clean, and not a duplicate -> approve without asking.
    #    The duplicate check has to happen here too, not just in the agent,
    #    or the fast path becomes the hole in the fence.
    if total <= config.AUTO_APPROVE_LIMIT and not result.issues:
        dup = tools.check_duplicate(
            conn, inv.vendor_name or "", inv.invoice_number or "",
            current_image=result.image,
        )
        if not dup.get("duplicate"):
            return AgentDecision(
                image=result.image, decision="approve", source="policy",
                reason=(f"{total:,.2f} SAR, under the "
                        f"{config.AUTO_APPROVE_LIMIT:,} SAR threshold, no "
                        f"validation findings, not a duplicate."),
            )

    return None            # the middle band -- the agent's job


# --------------------------------------------------------------------------
# What the model is shown
# --------------------------------------------------------------------------
def describe(result: Result) -> str:
    """
    Turn a Result into the text the model reads.

    Only the validated fields and the findings go in. Not the raw
    extraction, not the confidence internals -- extra detail is extra
    opportunity for the model to latch onto the wrong thing.
    """
    inv = result.invoice
    lines = [
        f"Document:        {result.image}",
        f"Supplier:        {inv.vendor_name}",
        f"Supplier VAT:    {inv.vendor_vat_number}",
        f"Buyer:           {inv.buyer_name}",
        f"Invoice number:  {inv.invoice_number}",
        f"Issue date:      {inv.issue_date}",
        f"Subtotal:        {inv.subtotal:,.2f} {inv.currency}",
        f"VAT ({inv.vat_rate:.0%}):       {inv.vat_amount:,.2f} {inv.currency}",
        f"Total:           {inv.total_amount:,.2f} {inv.currency}",
        f"Extraction confidence: {inv.confidence:.2f}",
    ]
    if result.issues:
        lines.append("\nValidation findings:")
        lines += [f"  [{i.severity}] {i.rule}: {i.detail}" for i in result.issues]
    else:
        lines.append("\nValidation findings: none")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def run_agent(result: Result, conn) -> AgentDecision:
    """
    Hand one invoice to Claude with the tools and let it decide.

    The loop is the whole agent: send the conversation, see what tool Claude
    asks for, run it, append the answer, send again. Stop when a decision
    tool comes back.
    """
    forced = policy_check(result, conn)
    if forced is not None:
        return forced

    client = anthropic.Anthropic(api_key=config.require_api_key())
    messages = [{"role": "user", "content": describe(result)}]
    out = AgentDecision(image=result.image, decision="escalate",
                        source="agent", reason="")

    for turn in range(1, MAX_TURNS + 1):
        out.turns = turn
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=tools.SCHEMAS,
            messages=messages,
        )
        out.input_tokens += response.usage.input_tokens
        out.output_tokens += response.usage.output_tokens

        calls = [b for b in response.content
                 if getattr(b, "type", None) == "tool_use"]

        # No tool call means Claude answered in prose instead of deciding.
        # That is a failure of the run, not a decision -- escalate it and
        # keep the text so you can see what it said instead.
        if not calls:
            said = " ".join(b.text for b in response.content
                            if getattr(b, "type", None) == "text").strip()
            out.reason = (f"Agent did not call a decision tool. It said: "
                          f"{said[:200] or '(nothing)'}")
            return out

        # Claude's turn goes into the history verbatim -- including the tool
        # request. The API requires the request and its result to pair up.
        messages.append({"role": "assistant", "content": response.content})

        results_block = []
        decided: AgentDecision | None = None

        for call in calls:
            out.trace.append(call.name)
            tool_out = tools.run_tool(conn, call.name, call.input,
                                      current_image=result.image)

            # tool_use_id is how an answer finds its question. Claude can
            # ask for two tools at once; without the id it would not know
            # which result belongs to which request.
            results_block.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(tool_out, ensure_ascii=False),
            })

            if call.name in tools.DECISION_TOOLS and "error" not in tool_out:
                decided = AgentDecision(
                    image=result.image,
                    decision=tool_out["decision"],
                    reason=tool_out["reason"],
                    source="agent", trace=out.trace, turns=turn,
                    input_tokens=out.input_tokens,
                    output_tokens=out.output_tokens,
                )

        if decided is not None:
            return veto(decided, result)

        messages.append({"role": "user", "content": results_block})

    out.reason = f"No decision after {MAX_TURNS} turns."
    return out


def veto(decision: AgentDecision, result: Result) -> AgentDecision:
    """
    The last guardrail: the model's answer still has to obey policy.

    An approval on an invoice carrying a validation finding gets overridden
    to escalate. The original reasoning is kept, because a model trying to
    approve something it should not is exactly the thing you want visible
    in the record rather than quietly discarded.
    """
    if decision.decision == "approve" and result.issues:
        found = "; ".join(f"{i.rule}" for i in result.issues)
        return AgentDecision(
            image=decision.image, decision="escalate", source="agent",
            reason=(f"Approval overridden by policy -- validation findings "
                    f"present ({found}). Agent had said: {decision.reason}"),
            trace=decision.trace, turns=decision.turns,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            overridden="approve",
        )
    return decision


# --------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python src/agent.py <image name, e.g. dev_000>")

    path = find_image(sys.argv[1])
    result = process(path)

    conn = connect()
    d = run_agent(result, conn)
    conn.close()

    print(f"\n{'=' * 60}")
    print(f"  {d.image}")
    print(f"{'=' * 60}")
    print(f"  pipeline   {result.status}")
    if result.issues:
        for i in result.issues:
            print(f"             [{i.severity}] {i.rule}")
    print(f"  decided by {d.source}")
    print(f"  decision   {d.decision.upper()}")
    if d.overridden:
        print(f"  overrode   {d.overridden}")
    print(f"  reason     {d.reason}")
    if d.source == "agent":
        print(f"  tools      {' -> '.join(d.trace) or '(none)'}")
        print(f"  turns      {d.turns}")
        print(f"  tokens     in {d.input_tokens:,}  out {d.output_tokens:,}"
              f"   ${d.cost:.4f}")
    else:
        print(f"  tools      (none -- settled in code, no API call)")
    print()


if __name__ == "__main__":
    main()
