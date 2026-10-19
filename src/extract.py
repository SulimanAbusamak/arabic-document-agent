"""
Send an invoice image to Claude and get back the raw extracted fields.

    python src/extract.py                    # one dev invoice
    python src/extract.py dev_004.png        # a specific one

Everything is cached to disk. The same image + model + prompt version is
only ever paid for once, however many times you run this.
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

import anthropic

import config
import prompts


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cache_key(image_path: Path, model: str) -> str:
    """
    A unique name for one (image CONTENTS, model, prompt) combination.

    Note it hashes the image's BYTES, not its filename. An earlier version
    used the name, which meant regenerating the invoices produced identical
    keys and every image silently returned a stale result -- a cache that
    never invalidates is worse than no cache, because it fails quietly.

    The prompt version is in there for the same reason: change the prompt
    and you get a different key, so you are never reading results produced
    by an older prompt.
    """
    image_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
    raw = f"{image_hash}|{model}|{prompts.PROMPT_VERSION}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{image_path.stem}_{prompts.PROMPT_VERSION}_{digest}.json"

def load_cached(key: str) -> dict | None:
    path = config.CACHE_DIR / key
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cached(key: str, payload: dict) -> None:
    path = config.CACHE_DIR / key
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def encode_image(image_path: Path) -> str:
    """Read an image file and base64-encode it for the API."""
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def reply_text(response) -> str:
    """
    Pull the text out of a response.

    response.content is a LIST of blocks, and they are not all text. Models
    with thinking enabled (Haiku 5.5 by default) return a ThinkingBlock first
    and the actual answer after it. Indexing [0] blindly crashes.

    So: filter by block type instead of assuming a position.
    """
    texts = [block.text for block in response.content
             if getattr(block, "type", None) == "text"]
    if not texts:
        kinds = [getattr(b, "type", "?") for b in response.content]
        raise ValueError(f"no text block in response; got blocks: {kinds}")
    return "\n".join(texts)


def parse_json_reply(text: str) -> dict:
    """
    Pull a JSON object out of the model's reply.

    Models sometimes wrap JSON in ```json fences or add a sentence before it,
    even when told not to. Rather than failing, find the outermost braces.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in reply:\n{text[:300]}")
    return json.loads(text[start:end + 1])


def extract(image_path: Path, model: str | None = None,
            use_cache: bool = True) -> dict:
    """
    Extract fields from one invoice image.

    Returns a dict with the parsed fields plus metadata about the call
    (which model, how many tokens, whether it came from cache).
    """
    model = model or config.MODEL
    key = cache_key(image_path, model)

    if use_cache:
        cached = load_cached(key)
        if cached is not None:
            cached["_from_cache"] = True
            return cached

    client = anthropic.Anthropic(api_key=config.require_api_key())

    response = client.messages.create(
        model=model,
        max_tokens=config.MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                # Image before text: the docs note Claude works best this way.
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encode_image(image_path),
                    },
                },
                {"type": "text", "text": prompts.EXTRACTION_PROMPT},
            ],
        }],
    )
    # Detect truncation explicitly. Without this a cut-off reply looks like
    # a JSON parsing failure, and you go hunting in the wrong place.
    if response.stop_reason == "max_tokens":
        raise ValueError(
            f"reply hit the {config.MAX_TOKENS}-token ceiling and was cut "
            f"off mid-JSON -- raise config.MAX_TOKENS"
        )

    reply = reply_text(response)
    fields = parse_json_reply(reply)

    result = {
        "fields": fields,
        "_meta": {
            "image": image_path.name,
            "model": model,
            "prompt_version": prompts.PROMPT_VERSION,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "_raw_reply": reply,
        "_from_cache": False,
    }

    save_cached(key, result)
    return result


# --------------------------------------------------------------------------
def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "dev_000.png"

    # look in dev first, then test
    for folder in (config.DEV_DIR, config.TEST_DIR):
        image_path = folder / name
        if image_path.exists():
            break
    else:
        raise SystemExit(f"{name} not found in {config.DEV_DIR} or {config.TEST_DIR}")

    print(f"image : {image_path.name}")
    result = extract(image_path)

    meta = result["_meta"]
    source = "cache" if result["_from_cache"] else "API"
    print(f"model : {meta['model']}  (from {source})")
    if not result["_from_cache"]:
        cost = (meta["input_tokens"] * 0.10 + meta["output_tokens"] * 0.50) / 1_000_000
        print(f"tokens: in {meta['input_tokens']}  out {meta['output_tokens']}"
              f"  (~${cost:.5f})")

    print("\nextracted:")
    print(json.dumps(result["fields"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
