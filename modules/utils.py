"""
Utility helpers for EPC app:
- value comparison (numeric, fuzzy string)
- normalization helpers for attribute keys and units
- vendor info extraction (regex heuristics + OpenAI fallback)

These functions are pure-Python and intended to be reused by TBE/TQ generators and UI.
"""
from typing import Tuple, Optional, Dict, Any
import re
import json
import os
from datetime import datetime

# fuzzy string match: prefer rapidfuzz if available
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    from difflib import SequenceMatcher
    _HAS_RAPIDFUZZ = False

# Attempt to import OpenAI client for vendor info fallback; if not available, we still work with regex heuristics
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
try:
    from openai import OpenAI as _OpenAIClient
    _HAS_OPENAI = bool(OPENAI_API_KEY)
except Exception:
    _OpenAI = None
    _HAS_OPENAI = False

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Basic unit mapping and simple conversion functions (expandable)
_unit_aliases = {
    'mm': 'mm', 'millimetre': 'mm', 'millimetres': 'mm', 'millimeter': 'mm', 'millimeters': 'mm',
    'in': 'in', 'inch': 'in', 'inches': 'in',
    'psi': 'psi', 'bar': 'bar',
    'degc': 'c', 'c': 'c', '°c': 'c', 'degf': 'f', 'f': 'f', '°f': 'f'
}

# very small converter: mm<->in, bar<->psi, c<->f
def _convert_units(value: float, from_u: str, to_u: str) -> Optional[float]:
    if from_u == to_u:
        return value
    if from_u == 'mm' and to_u == 'in':
        return value / 25.4
    if from_u == 'in' and to_u == 'mm':
        return value * 25.4
    if from_u == 'bar' and to_u == 'psi':
        return value * 14.503773773
    if from_u == 'psi' and to_u == 'bar':
        return value / 14.503773773
    if from_u == 'c' and to_u == 'f':
        return value * 9/5 + 32
    if from_u == 'f' and to_u == 'c':
        return (value - 32) * 5/9
    return None


def _now_iso():
    return datetime.utcnow().isoformat() + 'Z'


def is_number(s: Any) -> bool:
    try:
        if s is None:
            return False
        return bool(NUM_RE.search(str(s)))
    except Exception:
        return False


def parse_number(s: Any) -> Optional[float]:
    if s is None:
        return None
    m = NUM_RE.search(str(s))
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None


def _extract_unit(s: str) -> Optional[str]:
    if not s:
        return None
    # common pattern: number + space + unit (e.g., 10 mm, 5in, 14.5 psi)
    m = re.search(r"\b([a-zA-Z°%/]+)\b", s.replace('\u00B0', ''))
    if not m:
        return None
    u = m.group(1).lower().strip().rstrip('.')
    return _unit_aliases.get(u, u)


def normalize_key(k: str) -> str:
    """Normalize attribute keys to a canonical lowercase form (remove punctuation, collapse spaces).
    This helps mapping datasheet keys to vendor keys.
    """
    if not k:
        return ''
    k2 = re.sub(r"[^0-9a-zA-Z ]+", ' ', str(k)).strip().lower()
    k2 = re.sub(r"\s+", ' ', k2)
    return k2


def _fuzzy_ratio(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        try:
            return fuzz.ratio(a, b)
        except Exception:
            pass
    # fallback
    try:
        return SequenceMatcher(None, a, b).ratio() * 100.0
    except Exception:
        return 0.0


def compare_values(ds_val: Any, v_val: Any, rel_tol: float = 0.05, fuzzy_partial_threshold: int = 60) -> Tuple[str, float]:
    """Compare datasheet value and vendor value.
    Returns ("match"|"partial"|"non_compliant", score 0..1)
    Logic:
      - if numeric on both sides: compare numbers (convert simple units if detected)
      - if strings: exact match or fuzzy match
      - vendor blank -> non_compliant
    """
    # None/empty handling
    if (ds_val is None or str(ds_val).strip() == '') and (v_val is None or str(v_val).strip() == ''):
        return ('match', 1.0)
    if v_val is None or str(v_val).strip() == '':
        return ('non_compliant', 0.0)

    ds_str = str(ds_val).strip()
    v_str = str(v_val).strip()

    # numeric handling
    if is_number(ds_str) and is_number(v_str):
        a = parse_number(ds_str)
        b = parse_number(v_str)
        # try to detect units and convert if possible
        ds_unit = _extract_unit(ds_str)
        v_unit = _extract_unit(v_str)
        if ds_unit and v_unit and ds_unit != v_unit:
            conv = _convert_units(b, v_unit, ds_unit)
            if conv is not None:
                b = conv
        if a is None or b is None:
            # fallback to fuzzy string
            score = _fuzzy_ratio(ds_str.lower(), v_str.lower()) / 100.0
            if score >= 0.85:
                return ('match', score)
            elif score >= fuzzy_partial_threshold/100.0:
                return ('partial', score)
            else:
                return ('non_compliant', score)
        # numeric compare with relative tolerance
        tol = max(1e-9, rel_tol * abs(a) if abs(a) > 1e-9 else rel_tol)
        if abs(a - b) <= tol:
            return ('match', 1.0)
        elif abs(a - b) <= 2 * tol:
            return ('partial', 0.5)
        else:
            return ('non_compliant', max(0.0, 1 - min(abs(a - b) / max(abs(a), abs(b), 1e-9), 1.0)))

    # textual handling
    ds_norm = re.sub(r"\s+", ' ', ds_str.strip().lower())
    v_norm = re.sub(r"\s+", ' ', v_str.strip().lower())
    if ds_norm == v_norm:
        return ('match', 1.0)
    score = _fuzzy_ratio(ds_norm, v_norm) / 100.0
    if score >= 0.85:
        return ('match', score)
    elif score >= fuzzy_partial_threshold/100.0:
        return ('partial', score)
    else:
        return ('non_compliant', score)


# Vendor extraction heuristics
_VENDOR_NAME_PATTERNS = [
    re.compile(r"^\s*(?:Vendor|Supplier|Company|From)[:\-\s]+(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(.+?)\s*\n\s*(?:Address|Contact|Phone|Email)[:\-]", re.IGNORECASE | re.MULTILINE)
]
_ADDRESS_PATTERNS = [
    # naive address heuristics: lines containing numbers + street words or postal code patterns
    re.compile(r"\d{1,5}\s+[^,\n]+,?\s*[^,\n]+"),
    re.compile(r"\b\d{5}(?:-\d{4})?\b"),
]


def extract_vendor_info(text: str) -> Dict[str, str]:
    """Attempt to extract vendor name and address using regex heuristics.
    If environment OPENAI_API_KEY present and heuristics fail, optionally call OpenAI to extract.

    Returns: {"vendor_name":"...", "vendor_address":"...", "source":"heuristic"|"openai"|"none"}
    """
    if not text:
        return {'vendor_name': '', 'vendor_address': '', 'source': 'none'}

    # Heuristic name
    name = ''
    addr = ''
    for p in _VENDOR_NAME_PATTERNS:
        m = p.search(text)
        if m:
            name_candidate = m.group(1).strip()
            # discard too-long lines
            if 1 <= len(name_candidate) <= 200:
                name = name_candidate.split('\n')[0].strip()
                break
    # Heuristic address
    for p in _ADDRESS_PATTERNS:
        m = p.search(text)
        if m:
            addr = m.group(0).strip()
            break

    if name or addr:
        return {'vendor_name': name or '', 'vendor_address': addr or '', 'source': 'heuristic'}

    # OpenAI fallback (if available)
    if _HAS_OPENAI and OPENAI_API_KEY:
        try:
            client = _OpenAIClient(api_key=OPENAI_API_KEY)
            prompt = (
                "You are a strict JSON extractor. From the following document, extract a plausible vendor/company name and postal address.\n"
                "Return ONLY valid JSON with keys 'vendor_name' and 'vendor_address'. If a value cannot be found, return empty string for it.\n\n"
                "DOCUMENT:\n" + text[:4000]
            )
            # Use a simple chat/completion call shape - adapt if client differs
            resp = client.chat.completions.create(
                model='gpt-4o-mini' if os.environ.get('OPENAI_MODEL') is None else os.environ.get('OPENAI_MODEL'),
                messages=[
                    {"role": "system", "content": "You extract vendor name and address as JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            # parse response text
            content = ''
            if resp and getattr(resp, 'choices', None):
                # try to extract the assistant content field
                try:
                    content = resp.choices[0].message['content']
                except Exception:
                    content = str(resp)
            # attempt to find JSON blob
            jmatch = re.search(r"\{.*\}", content, re.DOTALL)
            if jmatch:
                j = jmatch.group(0)
                parsed = json.loads(j)
                return {'vendor_name': parsed.get('vendor_name', ''), 'vendor_address': parsed.get('vendor_address', ''), 'source': 'openai'}
        except Exception:
            pass

    return {'vendor_name': '', 'vendor_address': '', 'source': 'none'}


if __name__ == '__main__':
    # quick self-tests
    print(compare_values('10 mm', '0.39 in'))
    print(compare_values('SS316', 'ss316l'))
    print(extract_vendor_info('Vendor: Acme Pumps\nAddress: 123 Industrial Park, City X, 560001'))
