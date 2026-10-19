"""
Raw material for generating realistic Saudi tax invoices.

Everything here is invented. The VAT numbers are structurally valid
(15 digits, start and end with 3, per ZATCA's TIN format) but do not
belong to any real company.
"""

import random
import hashlib

# ---------------------------------------------------------------------------
# Vendors: (arabic_name, english_name)
# A mix of مؤسسة (establishment), شركة (company) and مجموعة (group),
# which is what you actually see on Saudi invoices.
# ---------------------------------------------------------------------------
VENDORS = [
    ("مؤسسة النور للتوريدات",              "Al Noor Supplies Est."),
    ("شركة الراجحي للمقاولات",             "Al Rajhi Contracting Co."),
    ("مؤسسة الخليج للتجارة",               "Gulf Trading Est."),
    ("شركة البيان لتقنية المعلومات",        "Al Bayan Information Technology Co."),
    ("مجموعة الفيصل التجارية",             "Al Faisal Commercial Group"),
    ("مؤسسة الشرق الأوسط للمعدات",         "Middle East Equipment Est."),
    ("شركة واحة الرياض للخدمات",           "Riyadh Oasis Services Co."),
    ("مؤسسة الديار للمواد الغذائية",        "Al Diyar Foodstuff Est."),
    ("شركة المراعي الحديثة للنقل",          "Modern Marai Transport Co."),
    ("مؤسسة ابن سينا للمستلزمات الطبية",    "Ibn Sina Medical Supplies Est."),
    ("شركة الأفق للحلول الرقمية",           "Al Ufuq Digital Solutions Co."),
    ("مؤسسة جدة للتبريد والتكييف",          "Jeddah Cooling & AC Est."),
]

BUYERS = [
    ("شركة الفيصل التجارية المحدودة",       "Al Faisal Trading Co. Ltd."),
    ("مؤسسة الصفوة للاستثمار",             "Al Safwa Investment Est."),
    ("شركة نجد للصناعات",                  "Najd Industries Co."),
    ("مجموعة الحرمين القابضة",             "Al Haramain Holding Group"),
    ("شركة تبوك للتنمية الزراعية",          "Tabuk Agricultural Development Co."),
]

CITIES = [
    ("الرياض", "Riyadh"),
    ("جدة", "Jeddah"),
    ("الدمام", "Dammam"),
    ("الخبر", "Khobar"),
    ("مكة المكرمة", "Makkah"),
    ("المدينة المنورة", "Madinah"),
]

DISTRICTS = [
    ("حي العليا", "Al Olaya District"),
    ("حي الملز", "Al Malaz District"),
    ("حي السلامة", "Al Salamah District"),
    ("حي النخيل", "Al Nakheel District"),
    ("حي الروضة", "Al Rawdah District"),
]

# (arabic_description, english_description, unit_ar, unit_en, low_price, high_price)
ITEMS = [
    ("خدمات استشارية تقنية",      "Technical Consulting Services",  "ساعة",  "Hour",   150,  600),
    ("أجهزة حاسب محمول",          "Laptop Computers",               "قطعة",  "Unit",  2500, 6500),
    ("طابعة ليزر ملونة",           "Color Laser Printer",            "قطعة",  "Unit",  1200, 3400),
    ("أوراق تصوير A4",            "A4 Photocopy Paper",             "صندوق", "Box",     55,  120),
    ("خدمة صيانة دورية",           "Periodic Maintenance Service",   "شهر",   "Month",  800, 2500),
    ("كراسي مكتبية دوارة",         "Swivel Office Chairs",           "قطعة",  "Unit",   450, 1300),
    ("تركيب شبكات داخلية",         "Internal Network Installation",  "نقطة",  "Point",  180,  450),
    ("مستلزمات طبية معقمة",        "Sterile Medical Supplies",       "علبة",  "Pack",    90,  380),
    ("خدمات نقل وشحن",            "Transport & Shipping Services",  "رحلة",  "Trip",   600, 2200),
    ("اشتراك برمجيات سنوي",        "Annual Software Subscription",   "ترخيص", "License", 900, 4800),
    ("أحبار طابعات أصلية",         "Original Printer Toner",         "قطعة",  "Unit",   220,  780),
    ("وحدات تكييف سبليت",          "Split Air Conditioning Units",   "قطعة",  "Unit",  1800, 5200),
]

PAYMENT_TERMS = [
    ("نقداً", "Cash"),
    ("تحويل بنكي", "Bank Transfer"),
    ("آجل 30 يوم", "Net 30 Days"),
    ("شبكة", "Mada / Card"),
]

VAT_RATE = 0.15  # Saudi standard rate


def saudi_vat_number(rng: random.Random) -> str:
    """
    Generate a structurally valid Saudi TIN / VAT registration number.

    ZATCA format: exactly 15 digits, beginning with 3 and ending with 3.
    Positions 2-13 are the taxpayer's identifier and branch code.
    This produces the right SHAPE, not a real registered number.
    """
    middle = "".join(str(rng.randint(0, 9)) for _ in range(13))
    return "3" + middle + "3"


def stable_vat_number(name: str) -> str:
    """
    One permanent VAT number per company, derived from its name.

    A company has exactly one tax registration, for life. Drawing a random
    one per invoice -- which is what saudi_vat_number does -- gave every
    vendor a different tax ID on every invoice they sent, which made the
    vendors table useless and the agent flag a mismatch on almost everything.

    Hashing the name means the same company always produces the same number,
    across runs, without needing to store a lookup table. Still structurally
    valid ZATCA: 15 digits, starts with 3, ends with 3.
    """
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    middle = "".join(str(int(c, 16) % 10) for c in digest[:13])
    return "3" + middle + "3"


def commercial_register(rng: random.Random) -> str:
    """Saudi CR number: 10 digits, conventionally starting with 1, 2, 4, 5 or 7."""
    return str(rng.choice([1, 2, 4, 5, 7])) + "".join(
        str(rng.randint(0, 9)) for _ in range(9)
    )


def phone_number(rng: random.Random) -> str:
    """Saudi landline or mobile."""
    if rng.random() < 0.5:
        return f"+966 5{rng.randint(0, 9)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"
    return f"+966 1{rng.randint(1, 3)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"


# Arabic-Indic numerals. Some Saudi invoices print amounts in these
# instead of Western digits, which is a real headache for extraction.
ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def to_arabic_indic(text: str) -> str:
    """Convert Western digits in a string to Arabic-Indic digits."""
    return str(text).translate(ARABIC_INDIC)
