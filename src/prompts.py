"""
Prompts for invoice extraction.

Kept in their own file because the prompt is the thing you will change most
often. PROMPT_VERSION is part of the cache key, so bumping it forces fresh
API calls instead of silently reusing results from an older prompt.
"""

PROMPT_VERSION = "v5"

# v2: added vat_rate, read off the invoice
# v3: rule 6 on doubled Arabic-Indic digits (measured: no change, kept for
#     the no-fabrication constraint)
# v4: line items. Needed for semantic duplicate detection -- comparing what
#     was actually bought, not just the totals. Two invoices for the same
#     job under different numbers have different totals lines and nearly
#     identical item descriptions.
# v5: three gaps found on a real bilingual invoice from an Indian supplier:
#     the currency was hardcoded to SAR in this prompt, so a dollar invoice
#     came back as riyals; the document carried TWO tax rates (15% and 10%)
#     and the model had one field for it; and it had a discount line with
#     nowhere to go.

EXTRACTION_PROMPT = """You are reading an Arabic/English tax invoice (most likely a Saudi Arabian tax invoice) (فاتورة ضريبية).

Extract these fields and return ONLY a JSON object, with no other text:

{
  "vendor_name": "supplier/seller name in Arabic, exactly as printed",
  "vendor_vat_number": "supplier VAT/tax number",
  "buyer_name": "customer name in Arabic, exactly as printed",
  "buyer_vat_number": "customer VAT/tax number",
  "invoice_number": "invoice number",
  "issue_date": "issue date, exactly as printed",
  "subtotal": "amount before VAT, exactly as printed",
  "vat_rate": "the VAT percentage printed on the invoice, as a number (15 for 15%)",
  "vat_amount": "VAT amount, exactly as printed",
  "total_amount": "total including VAT, exactly as printed",
  "discount": "any discount applied before tax, exactly as printed, or null",
  "currency": "the currency code as printed: SAR, USD, AED, JOD...",
  "line_items": [
    {
      "description": "the item or service description, exactly as printed",
      "quantity": "quantity, exactly as printed",
      "unit_price": "unit price, exactly as printed",
      "tax_rate": "this row's tax percentage, as a number, or null",
      "tax_amount": "this row's tax amount, exactly as printed, or null",
      "line_total": "line total, exactly as printed"
    }
  ],
  "confidence": 0.0 to 1.0,
  "unreadable_fields": ["list of field names you could not read"]
}

Rules:

1. Transcribe values EXACTLY as they appear on the document. Do not convert
   anything. If the invoice shows ٢٦,٠٤١.٦٥ then return "٢٦,٠٤١.٦٥". If it
   shows a Hijri date like ١٦/٩/١٤٤٧ هـ then return that, not a Gregorian
   date. Conversion happens later in the pipeline.

2. If you cannot read a field, set it to null and add its name to
   unreadable_fields. Never guess a value.

3. confidence reflects how legible the document is overall:
   - 0.9-1.0: clear and fully legible
   - 0.6-0.8: readable but some fields uncertain
   - below 0.5: too blurry, skewed or damaged to trust

4. Arabic invoices are read right to left. The supplier (المورّد) is usually
   at the top; the customer (العميل) is the party billed.

5. For vat_rate, read the percentage actually printed on the document
   (ضريبة القيمة المضافة ١٥٪ -> 15). Do not assume a rate. If the invoice
   shows no rate at all, set it to null.

6. Repeated adjacent digits are easy to miss in Arabic-Indic numerals: ٣٣,
   ٢٢ and ٠٠ can look like a single digit at this resolution. Saudi VAT
   registration numbers are always 15 digits long, so count the digits you
   have transcribed for each VAT number. If you count 14, look again for a
   doubled digit you may have read as one.

   Do NOT add a digit to make the count reach 15. If you look again and
   still read 14 digits, return the 14 you can actually see. A short number
   that is honest can be caught downstream; an invented digit cannot.
   
7. Transcribe every row of the items table into line_items, in the order
   they appear. If the invoice has no itemised table, return an empty list.
   Descriptions matter more than the numbers here: they are what identifies
   the work that was billed.

8. An invoice may apply DIFFERENT tax rates to different rows -- 15% on one
   item and 10% on another is normal on a mixed supply. Read each row's own
   rate and tax amount into that row, and put the invoice's headline rate
   in vat_rate as usual.


Return the JSON object and nothing else."""