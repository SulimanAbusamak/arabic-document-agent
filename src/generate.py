"""
Generate synthetic Saudi tax invoices (images) + ground truth (JSON).

Run:
    python src/generate.py

Produces:
    data/invoices/dev/   25 invoices  -> iterate on these
    data/invoices/test/  15 invoices  -> DO NOT look at these while tuning prompts
    data/ground_truth.json

Design notes
------------
* Ground truth stores CANONICAL values: Western digits, ISO dates, floats.
  The invoice image may show Arabic-Indic digits or a Hijri date instead.
  That mismatch is deliberate -- normalising it is part of the extraction job.
* 70% of invoices are bilingual (templates A and B), 30% Arabic-only (C).
* The QR code holds a real ZATCA-style TLV payload, base64 encoded.
"""

import base64
import io
import json
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import qrcode
from hijridate import Gregorian
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

import data as D

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
ROOT = SRC_DIR.parent
OUT_DEV = ROOT / "data" / "invoices" / "dev"
OUT_TEST = ROOT / "data" / "invoices" / "test"
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"

N_DEV = 25
N_TEST = 15
SEED = 42  # fixed so the dataset is reproducible

ARABIC_MONTHS = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]


# --------------------------------------------------------------------------
# ZATCA QR payload
# --------------------------------------------------------------------------
def zatca_tlv(seller: str, vat_no: str, timestamp: str, total: str, vat: str) -> str:
    """
    Build a ZATCA-style QR payload.

    Real Saudi e-invoices encode 5 fields as TLV (tag-length-value) bytes,
    then base64 the result. Tags: 1=seller name, 2=VAT number,
    3=timestamp, 4=invoice total (incl. VAT), 5=VAT amount.
    """
    payload = b""
    for tag, value in enumerate([seller, vat_no, timestamp, total, vat], start=1):
        encoded = value.encode("utf-8")
        payload += bytes([tag, len(encoded)]) + encoded
    return base64.b64encode(payload).decode("ascii")


def qr_data_uri(payload: str) -> str:
    """Render a QR code and return it as an inline base64 data URI."""
    qr = qrcode.QRCode(box_size=4, border=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def money(x: float) -> str:
    """Two decimals with a thousands separator: 12,345.60"""
    return f"{x:,.2f}"


def fmt_date(d: date, style: str) -> str:
    """
    Render a date the way a real invoice might.

    Both Hijri and Gregorian are widely used in Saudi commerce and the
    regulations don't mandate either, so the test set mixes them.
    """
    if style == "iso":
        return d.strftime("%Y-%m-%d")
    if style == "slash":
        return d.strftime("%d/%m/%Y")
    if style == "arabic_month":
        return f"{d.day} {ARABIC_MONTHS[d.month - 1]} {d.year}"
    if style == "hijri":
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        return f"{h.day}/{h.month}/{h.year} هـ"
    raise ValueError(style)


# --------------------------------------------------------------------------
# One invoice record
# --------------------------------------------------------------------------
def build_record(rng: random.Random, index: int) -> dict:
    """Create the data for a single invoice: truth + display strings."""

    vendor_ar, vendor_en = rng.choice(D.VENDORS)
    buyer_ar, buyer_en = rng.choice(D.BUYERS)
    city_ar, city_en = rng.choice(D.CITIES)
    buyer_city_ar, _ = rng.choice(D.CITIES)
    district_ar, _ = rng.choice(D.DISTRICTS)
    payment_ar, payment_en = rng.choice(D.PAYMENT_TERMS)

    vendor_vat = D.saudi_vat_number(rng)
    buyer_vat = D.saudi_vat_number(rng)
    vendor_cr = D.commercial_register(rng)
    vendor_phone = D.phone_number(rng)

    invoice_number = f"INV-{rng.randint(2025, 2026)}-{rng.randint(1, 9999):04d}"

    issue = date(2026, 1, 1) + timedelta(days=rng.randint(0, 280))
    supply = issue - timedelta(days=rng.randint(0, 10))

    # ---- line items -------------------------------------------------------
    items = []
    subtotal = 0.0
    for _ in range(rng.randint(1, 5)):
        desc_ar, desc_en, unit_ar, unit_en, lo, hi = rng.choice(D.ITEMS)
        qty = rng.randint(1, 20)
        unit_price = round(rng.uniform(lo, hi), 2)
        line_net = round(qty * unit_price, 2)
        line_vat = round(line_net * D.VAT_RATE, 2)
        line_total = round(line_net + line_vat, 2)
        subtotal += line_net
        items.append({
            "desc_ar": desc_ar, "desc_en": desc_en,
            "unit_ar": unit_ar, "unit_en": unit_en,
            "qty": qty,
            "unit_price_f": unit_price,
            "line_net_f": line_net,
            "line_vat_f": line_vat,
            "line_total_f": line_total,
        })

    subtotal = round(subtotal, 2)
    vat_amount = round(subtotal * D.VAT_RATE, 2)
    total_amount = round(subtotal + vat_amount, 2)

    # ---- presentation choices --------------------------------------------
    # 70% bilingual (A/B), 30% Arabic-only (C)
    template = rng.choices(["invoice_a", "invoice_b", "invoice_c"],
                           weights=[35, 35, 30])[0]
    arabic_only = template == "invoice_c"

    # Arabic-Indic digits appear on ~35% of the Arabic-only invoices
    use_indic = arabic_only and rng.random() < 0.55
    date_style = rng.choices(
        ["iso", "slash", "arabic_month", "hijri"],
        weights=[30, 35, 20, 15],
    )[0]

    qr_payload = zatca_tlv(
        seller=vendor_ar,
        vat_no=vendor_vat,
        timestamp=f"{issue.isoformat()}T10:30:00Z",
        total=f"{total_amount:.2f}",
        vat=f"{vat_amount:.2f}",
    )

    # ---- template context -------------------------------------------------
    def disp(value: str) -> str:
        """Apply Arabic-Indic digits if this invoice uses them."""
        return D.to_arabic_indic(value) if use_indic else str(value)

    ctx_items = []
    for i, it in enumerate(items, start=1):
        ctx_items.append({
            **it,
            "unit_price": money(it["unit_price_f"]),
            "line_net": money(it["line_net_f"]),
            "line_vat": money(it["line_vat_f"]),
            "line_total": money(it["line_total_f"]),
            # Arabic-only template uses the *_display variants
            "idx_display": disp(i),
            "qty_display": disp(it["qty"]),
            "unit_price_display": disp(money(it["unit_price_f"])),
            "line_net_display": disp(money(it["line_net_f"])),
            "line_vat_display": disp(money(it["line_vat_f"])),
            "line_total_display": disp(money(it["line_total_f"])),
        })

    issue_display = fmt_date(issue, date_style)
    supply_display = fmt_date(supply, date_style)

    context = {
        "vendor_ar": vendor_ar, "vendor_en": vendor_en,
        "buyer_ar": buyer_ar, "buyer_en": buyer_en,
        "vendor_city_ar": city_ar, "buyer_city_ar": buyer_city_ar,
        "vendor_district": district_ar,
        "vendor_vat": vendor_vat, "buyer_vat": buyer_vat,
        "vendor_cr": vendor_cr, "vendor_phone": vendor_phone,
        "invoice_number": invoice_number,
        "issue_date_display": disp(issue_display),
        "supply_date_display": disp(supply_display),
        "payment_ar": payment_ar, "payment_en": payment_en,
        "items": ctx_items,
        "subtotal": money(subtotal),
        "vat_amount": money(vat_amount),
        "total_amount": money(total_amount),
        "currency_label": "ر.س" if arabic_only else "SAR",
        "qr_data_uri": qr_data_uri(qr_payload),
        # Arabic-only template variants
        "vendor_vat_display": disp(vendor_vat),
        "buyer_vat_display": disp(buyer_vat),
        "vendor_cr_display": disp(vendor_cr),
        "vendor_phone_display": disp(vendor_phone),
        "invoice_number_display": disp(invoice_number),
        "subtotal_display": disp(money(subtotal)),
        "vat_amount_display": disp(money(vat_amount)),
        "total_amount_display": disp(money(total_amount)),
    }

    # ---- ground truth (canonical form, always Western digits + ISO dates) --
    truth = {
        "invoice_type": "فاتورة ضريبية",
        "vendor_name": vendor_ar,
        "vendor_vat_number": vendor_vat,
        "buyer_name": buyer_ar,
        "buyer_vat_number": buyer_vat,
        "invoice_number": invoice_number,
        "issue_date": issue.isoformat(),
        "supply_date": supply.isoformat(),
        "subtotal": subtotal,
        "vat_amount": vat_amount,
        "total_amount": total_amount,
        "currency": "SAR",
        "line_item_count": len(items),
        # metadata -- useful for slicing your evaluation results later
        "_meta": {
            "template": template,
            "bilingual": not arabic_only,
            "arabic_indic_digits": use_indic,
            "date_style": date_style,
        },
    }

    return {"template": template, "context": context, "truth": truth}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def main() -> None:
    rng = random.Random(SEED)
    env = Environment(loader=FileSystemLoader(SRC_DIR / "templates"))

    for folder in (OUT_DEV, OUT_TEST):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    records = [build_record(rng, i) for i in range(N_DEV + N_TEST)]
    ground_truth = {}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 794, "height": 1123},
                                device_scale_factor=2)

        for i, rec in enumerate(records):
            split = "dev" if i < N_DEV else "test"
            out_dir = OUT_DEV if split == "dev" else OUT_TEST
            filename = f"{split}_{i:03d}.png"

            html = env.get_template(rec["template"] + ".html").render(**rec["context"])
            page.set_content(html, wait_until="networkidle")
            page.screenshot(path=str(out_dir / filename), full_page=True)

            rec["truth"]["_meta"]["split"] = split
            ground_truth[filename] = rec["truth"]
            print(f"  {filename}  ({rec['template']}, {split})")

        browser.close()

    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bilingual = sum(1 for v in ground_truth.values() if v["_meta"]["bilingual"])
    indic = sum(1 for v in ground_truth.values() if v["_meta"]["arabic_indic_digits"])
    hijri = sum(1 for v in ground_truth.values() if v["_meta"]["date_style"] == "hijri")

    print(f"\n{len(ground_truth)} invoices generated")
    print(f"  bilingual            : {bilingual}")
    print(f"  arabic-only          : {len(ground_truth) - bilingual}")
    print(f"  arabic-indic digits  : {indic}")
    print(f"  hijri dates          : {hijri}")
    print(f"  ground truth         : {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
