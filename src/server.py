"""
An HTTP door into the pipeline, so something other than a human can use it.

    python src/server.py            # listens on 8000

    curl -X POST http://localhost:8000/process \
         -H "Content-Type: application/json" \
         -d "{\"image\": \"test_026.png\"}"

WHY THIS EXISTS
---------------
review.py runs the pipeline over a folder, once, when you type the command.
app.py runs it when a person drops a file in a browser. Neither can be
called by another program, and "another program" is the whole point of
putting this in a workflow: an invoice should be processed because it
arrived, not because somebody remembered to run a script.

n8n is that other program. It watches a folder and, when a file lands,
makes an HTTP request. So the pipeline needs an address.

WHAT IT IS NOT
--------------
This is not a production API. No authentication, no rate limiting, no
concurrency -- it handles one request at a time, which is correct here
because each request makes paid API calls and a queue is safer than a
stampede. It listens inside the Docker network only, where the sole
caller is n8n.

A real deployment would put FastAPI and a worker queue here. Saying that
plainly is better than pretending http.server is a web framework.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import embed
from database import connect, save_decision, save_result
from batch import result_to_dict

import config
import rules
from agent import run_agent
from database import connect, save_decision
from pipeline import find_image, process

PORT = 8000

# One connection for the life of the process.
#
# same_thread=False for the same reason app.py needs it: SQLite objects are
# bound to the thread that made them unless you say otherwise, and the
# server's handler may not be that thread.
_conn = connect(same_thread=False)
rules.init(_conn)


def to_dict(obj) -> dict:
    """
    AgentDecision -> plain JSON.

    asdict() if it is a dataclass, __dict__ otherwise, and anything that is
    still not JSON-serialisable (a Path, an enum) becomes a string. The
    caller is n8n, which wants text it can put in a spreadsheet, not Python
    objects.
    """
    raw = asdict(obj) if is_dataclass(obj) else dict(vars(obj))
    out = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool, type(None), list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

def embed_one(result) -> None:
    """
    Add one processed invoice to the vector store.

    Lifted from app.py, because an invoice that arrives through the
    workflow has to become history the same way one dropped in the browser
    does. Checked against the past but never added to it, and the
    reworded-rebill detection silently stops seeing anything n8n handled.
    """
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

def handle(image: str) -> dict:
    """
    One invoice, start to finish: find it, read it, decide, save.

    Exactly what review.py does per document, which is deliberate -- there
    should be one pipeline, not a slightly different one per entry point.
    A workflow that behaves differently from the CLI is a workflow nobody
    can debug.
    """
    path = Path(image)
    if not path.is_file():
        found = find_image(path.name)
        if not found:
            raise FileNotFoundError(f"no such image: {image}")
        path = found

    result = process(path)

    # Save the extracted record, not just the decision.
    #
    # The Review page joins decisions to invoices, and the duplicate check
    # reads the invoices table -- so a decision without its record is
    # invisible on screen AND invisible to the next invoice that arrives.
    # review.py never needed this because batch.py had already written
    # every row; an invoice arriving through the workflow has no such
    # head start.
    save_result(_conn, result_to_dict(result))
    if result.invoice is not None:
        embed_one(result)
    decision = run_agent(result, _conn)
    save_decision(_conn, decision)
    _conn.commit()

    out = to_dict(decision)
    out["image"] = path.name
    out["pipeline_status"] = result.status
    out["retried"] = result.retried
    out["recovered"] = result.recovered
    # Move it out of the inbox so the next scheduled run does not pay to
    # decide it again. The folder is the queue: in the inbox means not yet
    # done, and anything else means done. Nothing to remember, nothing to
    # get out of step after a crash.
    if path.parent.name == "inbox":
        done = path.parent.parent / "processed"
        done.mkdir(parents=True, exist_ok=True)
        path.rename(done / path.name)
        out["moved_to"] = f"processed/{path.name}"

    # The decision alone is not enough for anyone downstream. An email that
    # says "an invoice was held" without naming the supplier or the amount
    # makes the reader go and look it up, which defeats the point of sending
    # it.
    inv = result.invoice
    if inv is not None:
        d = inv.model_dump(mode="json")
        for k in ("vendor_name", "invoice_number", "issue_date",
                  "total_amount", "currency"):
            out[k] = d.get(k)
    return out


class Handler(BaseHTTPRequestHandler):

    def _send(self, code: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        # n8n and Docker both want a cheap way to ask "are you up?" that
        # does not cost an API call.
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "try POST /process"})

    def do_POST(self) -> None:
        if self.path != "/process":
            self._send(404, {"error": "try POST /process"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            image = body.get("image")
            if not image:
                self._send(400, {"error": "send {\"image\": \"name.png\"}"})
                return

            self._send(200, handle(image))

        except FileNotFoundError as e:
            self._send(404, {"error": str(e)})

        except Exception as e:
            # Never let the server die on one bad invoice. n8n gets a clear
            # failure for this document and keeps watching the folder; the
            # traceback goes to the log for you.
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args) -> None:
        # The default logs every request to stderr in Apache format, which
        # buries the pipeline's own output. One tidy line instead.
        print(f"  {self.address_string()} {fmt % args}")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"listening on 0.0.0.0:{port}")
    print(f"  GET  /health")
    print(f"  POST /process   {{\"image\": \"test_026.png\"}}\n")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
