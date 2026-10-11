"""
Create degraded copies of some invoices: blurry phone photos, bad scans.

Run (after generate.py):
    python src/degrade.py

These exist so you can test the REFUSAL path. A system that confidently
extracts wrong numbers from an unreadable scan is worse than useless in
finance -- it silently corrupts records. Your agent should decline these.

Two severities:
  phone_photo -- mild. Tilted, slightly soft. SHOULD still extract correctly.
  bad_scan    -- heavy. Genuinely unreadable. SHOULD be refused.

Both dev and test get both severities, so you can develop the refusal
logic against dev and still have untouched test cases to measure on.
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

SRC_DIR = Path(__file__).parent
ROOT = SRC_DIR.parent
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"

SEED = 7

# how many of each severity, per split
PLAN = {
    "dev":  {"phone_photo": 3, "bad_scan": 2},
    "test": {"phone_photo": 2, "bad_scan": 3},
}


def phone_photo(img: Image.Image, rng: random.Random) -> Image.Image:
    """Mild damage: a readable but imperfect phone snap. Should still extract."""
    img = img.rotate(rng.uniform(-2.5, 2.5), expand=True,
                     fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.2)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.1))
    # downscale then back up: loses detail the way a low-res photo does
    w, h = img.size
    scale = rng.uniform(0.45, 0.6)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img.resize((w, h), Image.LANCZOS)


def bad_scan(img: Image.Image, rng: random.Random) -> Image.Image:
    """Heavy damage: should trigger the agent's refusal path."""
    img = img.rotate(rng.uniform(-6, 6), expand=True,
                     fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.0, 3.2)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.45, 0.65))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 0.78))

    w, h = img.size
    scale = rng.uniform(0.22, 0.32)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = img.resize((w, h), Image.LANCZOS)

    # crop an edge off, as if the photo missed part of the page
    if rng.random() < 0.5:
        cut = int(h * rng.uniform(0.08, 0.16))
        img = img.crop((0, cut, w, h))
    return img


MODES = {"phone_photo": phone_photo, "bad_scan": bad_scan}


def main() -> None:
    rng = random.Random(SEED)
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

    for split, counts in PLAN.items():
        # Only pick from CLEAN originals. Without this filter, a second run
        # would degrade an already-degraded file (dev_004_deg.png also
        # starts with "dev_") and you'd get blur on top of blur.
        originals = sorted(
            k for k in gt
            if k.startswith(f"{split}_") and not gt[k]["_meta"].get("degraded")
        )

        # Sample everything in one go, so no invoice gets picked for two
        # severities and silently overwritten.
        picked = rng.sample(originals, sum(counts.values()))

        i = 0
        for mode, n in counts.items():
            for filename in picked[i:i + n]:
                folder = ROOT / "data" / "invoices" / split
                img = Image.open(folder / filename).convert("RGB")
                damaged = MODES[mode](img, rng)

                new_name = filename.replace(".png", "_deg.png")
                damaged.save(folder / new_name, quality=60)

                truth = json.loads(json.dumps(gt[filename]))  # deep copy
                truth["_meta"]["degraded"] = True
                truth["_meta"]["degradation"] = mode
                truth["_meta"]["expect_refusal"] = (mode == "bad_scan")
                truth["_meta"]["derived_from"] = filename
                gt[new_name] = truth

                print(f"  {new_name:24s} {mode}")
            i += n

    GROUND_TRUTH.write_text(json.dumps(gt, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    def count(split, key=None):
        items = [v for k, v in gt.items() if k.startswith(f"{split}_")]
        if key is None:
            return len(items)
        return sum(1 for v in items if v["_meta"].get(key))

    print(f"\n{'':6s} {'total':>6s} {'degraded':>9s} {'refusals':>9s}")
    for split in ("dev", "test"):
        print(f"{split:6s} {count(split):>6d} {count(split,'degraded'):>9d} "
              f"{count(split,'expect_refusal'):>9d}")
    print(f"\ntotal dataset: {len(gt)} images")


if __name__ == "__main__":
    main()