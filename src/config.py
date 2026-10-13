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

MAX_TOKENS = 2000


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
