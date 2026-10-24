"""
Central settings for the project.

Everything that might need changing lives here, so you change it in one
place instead of hunting through the codebase. Paths are worked out from
this file's location, so the project runs from any folder and inside Docker.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read .env into the environment. Must happen before os.getenv below.
load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
ROOT = SRC_DIR.parent

DATA_DIR = ROOT / "data"
INVOICES_DIR = DATA_DIR / "invoices"
DEV_DIR = INVOICES_DIR / "dev"
TEST_DIR = INVOICES_DIR / "test"
GROUND_TRUTH = DATA_DIR / "ground_truth.json"
REAL_DIR = DATA_DIR / "real"

CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Haiku is built for extraction and routing, and is ~20x cheaper than Sonnet.
# Start here; switch to MODEL_STRONG and compare on the dev set to find out
# whether the extra cost buys anything.
MODEL = "claude-haiku-5-5"
MODEL_STRONG = "claude-sonnet-5-5"

MAX_TOKENS = 4000
# --- retry on failure -----------------------------------------------------
# When the cheap model cannot read a document, try once more with a stronger
# one before giving up on it. A cheap-first cascade: Haiku reads everything,
# Sonnet only ever sees the few documents Haiku failed on.
#
# Enlarging the image was the obvious other lever and it is not available.
# These models look at images up to 2576 px on the long edge; our invoices
# are already 2246, so there is ~15% of headroom and anything bigger gets
# downscaled by the API before the model sees it. Cropping to the totals
# region would work -- fewer pixels of page, all of them useful -- but that
# is a bigger change.
RETRY_MODEL = MODEL_STRONG

# Per-million-token prices, per model. A single hardcoded pair was wrong the
# moment a second model entered the system -- and wrong in the dangerous
# direction, understating the bill.
#
# Note Haiku 5.5 is tiered: prompts over 100,000 tokens cost $0.50/$2.50
# instead. Our calls are ~5,000 tokens so this never applies, but an agent
# loop that ran long enough would cross it.
PRICING = {
    "claude-haiku-5-5":  {"in": 0.10, "out": 0.50},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00},
}


def call_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """What one API call cost, in dollars."""
    p = PRICING.get(model, PRICING[MODEL])
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000

# --------------------------------------------------------------------------
# Business rules
#
# Nothing here is hardcoded in the validation code -- change a number here
# and every check follows. That is what makes the system work for Jordan
# (16%) or a zero-rated invoice without touching any logic.
# --------------------------------------------------------------------------

# Rates we recognise. The rate is READ OFF each invoice, not assumed; this
# set is only a sanity check that the extracted rate is plausible.
#   0.00 zero-rated/exempt · 0.05 UAE · 0.15 Saudi · 0.16 Jordan
KNOWN_VAT_RATES = {0.00, 0.05, 0.15, 0.16}

# Fallback when an invoice does not print its rate at all.
DEFAULT_VAT_RATE = 0.15

# Saudi ZATCA tax identification numbers are 15 digits.
VAT_NUMBER_LENGTH = 15

# How far the arithmetic may be off before we call it wrong. Invoices round
# each line item, so the sum of the parts can differ from the printed total
# by a halala or two. More than this and a digit was misread.
MONEY_TOLERANCE = 0.02

# Below this, the document is treated as too illegible to trust.
CONFIDENCE_FLOOR = 0.50

# Approval thresholds, used by the agent on Day 4.
AUTO_APPROVE_LIMIT = 15_000      # known vendor, clean checks -> approve
ESCALATION_LIMIT = 120_000       # above this, a human always looks


def require_api_key() -> str:
    """
    Fail loudly and usefully if the key is missing.

    Without this you'd get a confusing error from deep inside the SDK.
    """
    if not API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found.\n"
            f"Create a file called .env in {ROOT} containing:\n"
            "    ANTHROPIC_API_KEY=sk-ant-...\n"
            "Then check it is listed in .gitignore."
        )
    return API_KEY
