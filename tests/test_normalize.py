"""
Tests for the normaliser.

    python tests/test_normalize.py     # no extra tools needed
    pytest                             # if you have pytest installed

No API calls, no files, no network. Runs in well under a second, so there
is no excuse not to run it after every change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalize import (  # noqa: E402
    normalize_arabic,
    normalize_record,
    parse_date,
    parse_money,
    to_western_digits,
)


# --------------------------------------------------------------------------
def test_western_digits():
    assert to_western_digits("٢٦,٠٤١.٦٥") == "26,041.65"
    assert to_western_digits("١٤٤٧") == "1447"
    assert to_western_digits("2026") == "2026"          # already western
    assert to_western_digits("ر.س ٤٢٠٠") == "ر.س 4200"  # letters untouched
    assert to_western_digits(None) is None


def test_money_basic():
    assert parse_money("19,270.50 SAR") == 19270.5
    assert parse_money("22,161.07") == 22161.07
    assert parse_money("4200") == 4200.0
    assert parse_money(19270.5) == 19270.5              # already a number


def test_money_arabic():
    assert parse_money("٢٦,٠٤١.٦٥") == 26041.65
    assert parse_money("٨٩١.٩٥") == 891.95
    assert parse_money("٤٢٠٠") == 4200.0


def test_money_currency_symbols():
    # ر.س contains a dot -- it must be stripped before deciding which
    # separator is the decimal point, or parsing breaks.
    assert parse_money("٢٦,٠٤١.٦٥ ر.س") == 26041.65
    assert parse_money("ر.س ٣,٣٩٦.٧٤") == 3396.74
    assert parse_money("1,500.00 SAR") == 1500.0


def test_money_european_style():
    # comma as the decimal separator
    assert parse_money("1.500,00") == 1500.0
    assert parse_money("1.500.000,00") == 1500000.0


def test_money_rejects_junk():
    assert parse_money("") is None
    assert parse_money("abc") is None
    assert parse_money(None) is None


# --------------------------------------------------------------------------
def test_date_gregorian():
    assert parse_date("2026-03-05") == "2026-03-05"
    assert parse_date("25/09/2026") == "2026-09-25"     # day first
    assert parse_date("٢٥/٠٩/٢٠٢٦") == "2026-09-25"


def test_date_hijri():
    # marked with هـ
    assert parse_date("16/9/1447 هـ") == "2026-03-05"
    assert parse_date("١٠/٩/١٤٤٧ هـ") == "2026-02-27"
    # unmarked, but the year is far too small to be Gregorian
    assert parse_date("16/9/1447") == "2026-03-05"


def test_date_arabic_month_name():
    assert parse_date("5 مارس 2026") == "2026-03-05"
    assert parse_date("25 سبتمبر 2026") == "2026-09-25"


def test_date_rejects_junk():
    assert parse_date("not a date") is None
    assert parse_date("45/99/2026") is None             # impossible day/month
    assert parse_date(None) is None


# --------------------------------------------------------------------------
def test_arabic_spelling_variants_match():
    """The same vendor spelled differently must produce the same key."""
    pairs = [
        ("مؤسسة النور", "مؤسسه النور"),        # ة vs ه
        ("شركة الأفق", "شركه الافق"),          # أ vs ا
        ("مـــؤسسة الديار", "مؤسسة الديار"),   # tatweel stretching
        ("مجموعة  الفيصل", "مجموعة الفيصل"),   # double space
    ]
    for a, b in pairs:
        assert normalize_arabic(a) == normalize_arabic(b), f"{a} != {b}"


def test_different_vendors_do_not_match():
    """Normalising must not be so aggressive that real names collide."""
    assert normalize_arabic("مؤسسة النور") != normalize_arabic("مؤسسة الديار")


# --------------------------------------------------------------------------
def test_full_record():
    """The real output from dev_000.png, straight from the API."""
    raw = {
        "vendor_name": "شركة الأفق للحلول الرقمية",
        "vendor_vat_number": "332181960013383",
        "buyer_name": "شركة الفيصل التجارية المحدودة",
        "buyer_vat_number": "390838637940263",
        "invoice_number": "INV-2026-8786",
        "issue_date": "16/9/1447 هـ",
        "subtotal": "19,270.50 SAR",
        "vat_amount": "2,890.57 SAR",
        "total_amount": "22,161.07 SAR",
        "currency": "SAR",
    }
    out = normalize_record(raw)

    assert out["subtotal"] == 19270.5
    assert out["vat_amount"] == 2890.57
    assert out["total_amount"] == 22161.07
    assert out["issue_date"] == "2026-03-05"
    # the name keeps its real spelling
    assert out["vendor_name"] == "شركة الأفق للحلول الرقمية"
    # and gains a flattened key for matching
    assert out["vendor_name_key"] == normalize_arabic(raw["vendor_name"])


def test_vat_arithmetic_holds_after_normalising():
    """
    subtotal + vat == total, to the halala.

    This is the check that catches a misread digit later: if the three
    numbers do not add up, one of them was read wrong.
    """
    out = normalize_record({
        "subtotal": "19,270.50", "vat_amount": "2,890.57",
        "total_amount": "22,161.07",
    })
    assert round(out["subtotal"] + out["vat_amount"], 2) == out["total_amount"]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}  {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}  {type(e).__name__}: {e}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
