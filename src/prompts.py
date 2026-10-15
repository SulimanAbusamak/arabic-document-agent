"""
Prompts for invoice extraction.

Kept in their own file because the prompt is the thing you will change most
often. PROMPT_VERSION is part of the cache key, so bumping it forces fresh
API calls instead of silently reusing results from an older prompt.
"""

PROMPT_VERSION = "v2"   # v2: added vat_rate, read off the invoice

EXTRACTION_PROMPT = """You are reading a Saudi Arabian tax invoice (فاتورة ضريبية).

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
  "currency": "SAR",
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

Return the JSON object and nothing else."""
