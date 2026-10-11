"""
Create degraded copies of some invoices: blurry phone photos, bad scans.

Run (after generate.py):
    python src/degrade.py

These exist so you can test the REFUSAL path. A system that confidently
extracts wrong numbers from an unreadable scan is worse than useless in
finance -- it silently corrupts records. Your agent should decline these.

Degraded files are written alongside the originals with a _deg suffix,
and registered in ground_truth.json with "_meta.degraded": true.
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

SRC_DIR = Path(__file__).parent
ROOT = SRC_DIR.parent
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"

SEED = 7
N_DEGRADE_DEV = 3
N_DEGRADE_TEST = 5


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


def main() -> None:
    rng = random.Random(SEED)
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

    dev_files = sorted(k for k in gt if k.startswith("dev_"))
    test_files = sorted(k for k in gt if k.startswith("test_"))

    targets = (
        [(f, "phone_photo") for f in rng.sample(dev_files, N_DEGRADE_DEV)]
        + [(f, "phone_photo") for f in rng.sample(test_files, 2)]
        + [(f, "bad_scan") for f in rng.sample(test_files, N_DEGRADE_TEST - 2)]
    )

    for filename, mode in targets:
        split = "dev" if filename.startswith("dev_") else "test"
        folder = ROOT / "data" / "invoices" / split
        src = folder / filename
        if not src.exists():
            continue

        img = Image.open(src).convert("RGB")
        damaged = phone_photo(img, rng) if mode == "phone_photo" else bad_scan(img, rng)

        new_name = filename.replace(".png", "_deg.png")
        damaged.save(folder / new_name, quality=60)

        truth = json.loads(json.dumps(gt[filename]))  # deep copy
        truth["_meta"]["degraded"] = True
        truth["_meta"]["degradation"] = mode
        # a bad_scan is not expected to be extractable -- the agent should refuse
        truth["_meta"]["expect_refusal"] = (mode == "bad_scan")
        truth["_meta"]["derived_from"] = filename
        gt[new_name] = truth

        print(f"  {new_name}  ({mode})")

    GROUND_TRUTH.write_text(json.dumps(gt, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    n_deg = sum(1 for v in gt.values() if v["_meta"].get("degraded"))
    n_refuse = sum(1 for v in gt.values() if v["_meta"].get("expect_refusal"))
    print(f"\n{n_deg} degraded invoices added ({n_refuse} expected to be refused)")
    print(f"total dataset: {len(gt)} images")


if __name__ == "__main__":
    main()
