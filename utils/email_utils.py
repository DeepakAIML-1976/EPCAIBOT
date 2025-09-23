# utils/email_utils.py

import re
import os
import json
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
except Exception:
    openai_client = None

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

def _heuristic_extract(text, filename=None):
    out = {"name": None, "address": "", "emails": [], "phones": []}
    emails = _EMAIL_RE.findall(text)
    out["emails"] = list(dict.fromkeys(emails))
    if filename and not out["name"]:
        base = filename.rsplit(".", 1)[0]
        out["name"] = base.replace("_", " ").replace("-", " ")
    m = re.search(r"(?:vendor|company|supplier)[:\-\s]+(.+)", text, re.I)
    if m and not out["name"]:
        out["name"] = m.group(1).strip()
    m2 = re.search(r"(?:address|addr|office)[\s:]+(.+)", text, re.I)
    if m2:
        out["address"] = m2.group(1).strip()
    return out

def _openai_extract(text):
    if not openai_client:
        return {"name": "", "address": "", "emails": []}
    prompt = (
        "Extract vendor name, address and emails from the text below. Return valid JSON "
        "with keys: name, address, emails (list). If not found, use empty strings or empty list.\n\n"
        + text[:20000]
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
            temperature=0.0,
        )
        out = resp.choices[0].message.content
    except Exception:
        return {"name": "", "address": "", "emails": []}
    # parse JSON-structure
    m = re.search(r"(\{[\s\S]*\})", out)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return {"name": parsed.get("name",""), "address": parsed.get("address",""), "emails": parsed.get("emails",[])}
        except Exception:
            return {"name":"", "address":"", "emails":[]}
    return {"name":"", "address":"", "emails":[]}

def extract_vendor_metadata(vendor_text, vendor_filename=None, prefer_openai=True):
    heur = _heuristic_extract(vendor_text, vendor_filename)
    if prefer_openai and openai_client:
        try:
            ai = _openai_extract(vendor_text)
            # merge heuristics and ai
            name = heur.get("name") or ai.get("name")
            address = heur.get("address") or ai.get("address")
            emails = heur.get("emails") or ai.get("emails")
            return {"name": name, "address": address, "emails": emails}
        except Exception:
            return heur
    return heur

def draft_tq_email(vendor_text, vendor_filename, tq_text, datasheet_ref=None, sender_name="EPC Team"):
    meta = extract_vendor_metadata(vendor_text, vendor_filename)
    to = meta["emails"][0] if meta["emails"] else ""
    subject = f"Technical Query regarding {datasheet_ref or 'Datasheet'}"
    salutation = f"Dear {meta['name']}" if meta.get("name") else "Dear Sir/Madam"
    address_block = (meta.get("address") + "\n\n") if meta.get("address") else ""
    body = (
        f"{address_block}{salutation},\n\n"
        f"Please find below a Technical Query (TQ) raised against your offer and our datasheet {datasheet_ref or ''}:\n\n"
        f"{tq_text}\n\n"
        "Kindly respond with clarifications and any supporting documents.\n\n"
        f"Regards,\n{sender_name}\n"
    )
    return {"vendor": meta, "email": {"to": to, "subject": subject, "body": body}}
