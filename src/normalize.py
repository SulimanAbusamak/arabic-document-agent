"""
Turn what Claude read off the invoice into canonical values.

Claude transcribes exactly what is printed: "٢٦,٠٤١.٦٥ ر.س", "16/9/1447 هـ".
Ground truth stores canonical values: 26041.65, "2026-03-05". This module
bridges the two.

Every function here is pure: same input, same output, no API calls, no files.
That is what makes it testable in a tenth of a second (see tests/).

Two different jobs live here, and mixing them up causes bugs:

  * CONVERSION  (parse_money, parse_date)
      Changes the value into a usable type. The result is what you store.

  * MATCHING    (normalize_arabic)
      Flattens spelling variants so two forms of the same name compare equal.
      The result is a lookup key, NEVER what you store -- it throws away
      information the real name had.
"""

import re
import unicodedata
from datetime import date

from hijridate import Hijri

# --------------------------------------------------------------------------
# Digits
# --------------------------------------------------------------------------

# Arabic-Indic (٠-٩) used across the Gulf, and Eastern Arabic-Indic (۰-۹)
# used in Persian/Urdu. Both turn up in scanned documents.
_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩" "۰۱۲۳۴۵۶۷۸۹",
    "0123456789" "0123456789",
)

# Arabic decimal separator (٫) and thousands separator (٬)
_SEPARATOR_MAP = str.maketrans("٫٬", ".,")


def to_western_digits(text: str) -> str:
    """
    '٢٦,٠٤١.٦٥' -> '26,041.65'

    Only touches digits and separators. Letters are left alone.
    """
    if text is None:
        return None
    return str(text).translate(_DIGIT_MAP).translate(_SEPARATOR_MAP)


# --------------------------------------------------------------------------
# Arabic text, for matching only
# --------------------------------------------------------------------------

# Harakat (fatha, kasra, shadda...) and the superscript alef. Optional in
# writing, so the same name may or may not carry them.
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
#                                                 ^ tatweel, the stretching
#                                                   character in مـــؤسسة

_LETTER_MAP = str.maketrans({
    # alef with any hamza or madda -> bare alef
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    # teh marbuta -> heh   (مؤسسة / مؤسسه)
    "ة": "ه",
    # alef maqsura -> yeh  (مستشفى / مستشفي)
    "ى": "ي",
    # hamza carriers
    "ؤ": "و", "ئ": "ي",
})


def normalize_arabic(text: str) -> str:
    """
    Flatten Arabic spelling variants so the same name compares equal.

        'مؤسسة النور'  -> 'موسسه النور'
        'مؤسسه النور'  -> 'موسسه النور'   <- same key, matches

    USE FOR COMPARING ONLY. Store the original spelling; this result has
    lost information and would look wrong on a report.
    """
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", str(text))
    text = _DIACRITICS.sub("", text)
    text = text.translate(_LETTER_MAP)
    # collapse runs of whitespace
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------

_NOT_NUMERIC = re.compile(r"[^\d.\-]")


def parse_money(text) -> float | None:
    """
    '19,270.50 SAR' -> 19270.5
    '٢٦,٠٤١.٦٥ ر.س' -> 26041.65
    '1.500,00'       -> 1500.0     (European style, comma as decimal)

    Returns None if there is no number in there at all.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    s = to_western_digits(text)

    # Strip everything that is not a digit or a separator FIRST.
    #
    # This has to come before the comma/dot logic below, because the Arabic
    # riyal symbol ر.س contains a dot. Leaving it in makes '٢٦,٠٤١.٦٥ ر.س'
    # look like its last separator is that dot, and parsing falls apart.
    s = re.sub(r"[^\d.,\-]", "", s)
    s = s.strip(".,")                    # drop separators left stranded at the ends

    # Decide what the comma means by looking at which separator comes LAST.
    # '19,270.50' -> dot is last   -> comma is thousands, drop it
    # '1.500,00'  -> comma is last -> comma is the decimal point
    last_dot = s.rfind(".")
    last_comma = s.rfind(",")
    if last_comma > last_dot:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")

    if not s or s in {".", "-", "-."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fix_rtl_order(text: str) -> str:
    """
    Undo right-to-left reversal in a reference like an invoice number.

        '1318-2026-INV' -> 'INV-2026-1318'

    Arabic renders right to left, so a Latin-prefixed reference printed on
    an Arabic invoice DISPLAYS with its segments in reverse order. A model
    reading the page visually transcribes that reversed order.

    Detection is deliberately narrow: only flip when the first segment is
    all digits AND the last is all letters, which is the signature of a
    reversed reference. A real date ('2026-03-05') or a dashed number
    ('332-181-960') is left alone because its last segment is numeric.
    """
    if not text or "-" not in text:
        return text
    parts = str(text).split("-")
    if len(parts) < 2:
        return text
    if parts[0].isdigit() and parts[-1].isalpha():
        return "-".join(reversed(parts))
    return text


def parse_rate(text) -> float | None:
    """
    A VAT percentage as printed -> a fraction.

        '15%'   -> 0.15
        '١٥٪'   -> 0.15      (Arabic percent sign)
        15      -> 0.15      (bare number, read as a percentage)
        0.15    -> 0.15      (already a fraction, left alone)
        '0%'    -> 0.0

    The 0-1 vs 1-100 ambiguity is resolved by size: nothing above 1 can be
    a fraction, and a real VAT rate is never 15000%.
    """
    if text is None:
        return None
    if isinstance(text, str):
        text = text.replace("٪", "%")
    value = parse_money(text)      # strips %, spaces, Arabic digits
    if value is None:
        return None
    if value > 1:                  # 15 means 15%
        value = value / 100
    return round(value, 4)


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_ARABIC_MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11,
    "ديسمبر": 12,
    # common alternates
    "كانون الثاني": 1, "شباط": 2, "اذار": 3, "نيسان": 4, "ايار": 5,
    "حزيران": 6, "تموز": 7, "اب": 8, "ايلول": 9, "تشرين الاول": 10,
    "تشرين الثاني": 11, "كانون الاول": 12,
}

# هـ / ه / هجري all mark a Hijri date
_HIJRI_MARK = re.compile(r"\b(هـ|هجري|AH)\b|هـ")


def _is_hijri(raw: str, year: int) -> bool:
    """
    Hijri if the document says so, or if the year is too small to be
    Gregorian. The Hijri year is currently ~1447; Gregorian ~2026.
    """
    if _HIJRI_MARK.search(raw):
        return True
    return 1300 <= year <= 1500


def parse_date(text) -> str | None:
    """
    Any printed date -> ISO 'YYYY-MM-DD'.

        '2026-03-05'      -> '2026-03-05'
        '05/03/2026'      -> '2026-03-05'   (day first)
        '16/9/1447 هـ'    -> '2026-03-05'   (Hijri converted)
        '٢٥/٠٩/٢٠٢٦'      -> '2026-09-25'
        '5 مارس 2026'     -> '2026-03-05'

    Returns None if nothing date-shaped is found.
    """
    if text is None:
        return None
    if isinstance(text, date):
        return text.isoformat()

    raw = str(text).strip()
    s = to_western_digits(raw)

    # --- named Arabic month: "5 مارس 2026" --------------------------------
    key = normalize_arabic(s)
    for name, month in _ARABIC_MONTHS.items():
        if name in key:
            nums = re.findall(r"\d+", key)
            if len(nums) >= 2:
                day, year = int(nums[0]), int(nums[-1])
                return _build(day, month, year, raw)

    # --- numeric: 2026-03-05, 05/03/2026, 16.9.1447 -----------------------
    nums = re.findall(r"\d+", s)
    if len(nums) < 3:
        return None
    a, b, c = (int(n) for n in nums[:3])

    if len(nums[0]) == 4:        # year first:  2026-03-05
        year, month, day = a, b, c
    else:                        # day first:   05/03/2026
        day, month, year = a, b, c

    return _build(day, month, year, raw)


def _build(day: int, month: int, year: int, raw: str) -> str | None:
    """Validate the parts, convert from Hijri if needed, return ISO."""
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    try:
        if _is_hijri(raw, year):
            return Hijri(year, month, day).to_gregorian().isoformat()
        return date(year, month, day).isoformat()
    except (ValueError, OverflowError):
        return None


# --------------------------------------------------------------------------
# Whole record
# --------------------------------------------------------------------------

_MONEY_FIELDS = ("subtotal", "vat_amount", "total_amount")
_RATE_FIELDS = ("vat_rate",)
_DATE_FIELDS = ("issue_date", "supply_date", "due_date")
_TEXT_FIELDS = ("vendor_name", "buyer_name")
_ID_FIELDS = ("vendor_vat_number", "buyer_vat_number", "invoice_number")


def normalize_line_items(items) -> list[dict]:
    """
    Clean each row of the items table.

    Without this the whole invoice dies. Claude transcribes exactly what is
    printed, so a quantity arrives as '٢٠' and a price as '٤٥,٠٠٠.٠٠' --
    and Pydantic cannot make a float out of either, so building the Invoice
    raises and a perfectly readable document becomes 'unusable'.

    Rows that are entirely empty are dropped. A row with an unreadable
    number keeps its description and loses the number: the description is
    what we actually want these for.
    """
    if not isinstance(items, list):
        return []

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = {
            "description": (" ".join(str(item["description"]).split())
                            if item.get("description") else None),
            "quantity": parse_money(item.get("quantity")),
            "unit_price": parse_money(item.get("unit_price")),
            "line_total": parse_money(item.get("line_total")),
        }
        if any(v is not None for v in row.values()):
            out.append(row)
    return out

def normalize_record(fields: dict) -> dict:
    """
    Run a whole extracted record through the right converter per field.

    Names keep their original spelling. A *_key version is added alongside,
    for matching against existing vendor records.
    """
    out = dict(fields)

    for f in _MONEY_FIELDS:
        if f in out:
            out[f] = parse_money(out[f])

    for f in _RATE_FIELDS:
        if f in out:
            out[f] = parse_rate(out[f])

    for f in _DATE_FIELDS:
        if f in out:
            out[f] = parse_date(out[f])

    for f in _ID_FIELDS:
        if f in out and out[f] is not None:
            # IDs are digits: westernise, then keep only digits and dashes
            s = to_western_digits(out[f])
            s = re.sub(r"[^\w\-]", "", s).strip()
            # a reference read off an RTL page may have arrived backwards
            out[f] = fix_rtl_order(s)

    for f in _TEXT_FIELDS:
        if f in out and out[f] is not None:
            out[f] = " ".join(str(out[f]).split())   # tidy whitespace only
            out[f + "_key"] = normalize_arabic(out[f])

    if "line_items" in out:
        out["line_items"] = normalize_line_items(out["line_items"])
        
    return out
