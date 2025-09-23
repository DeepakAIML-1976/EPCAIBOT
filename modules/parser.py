# modules/parser.py
import io
import os
from typing import Optional
import re

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

def _clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"\r\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def extract_text_from_pdf_bytes(b: bytes) -> str:
    text_parts = []
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        text_parts.append(t)
            return _clean_text("\n".join(text_parts))
        except Exception:
            pass
    if PdfReader:
        try:
            reader = PdfReader(io.BytesIO(b))
            for p in reader.pages:
                try:
                    t = p.extract_text()
                except Exception:
                    t = ""
                if t:
                    text_parts.append(t)
            return _clean_text("\n".join(text_parts))
        except Exception:
            pass
    return ""

def extract_text_from_docx_bytes(b: bytes) -> str:
    if docx:
        try:
            tmp = io.BytesIO(b)
            document = docx.Document(tmp)
            paragraphs = [p.text for p in document.paragraphs if p.text]
            return _clean_text("\n".join(paragraphs))
        except Exception:
            return ""
    return ""

def extract_text_from_excel_bytes(b: bytes) -> str:
    if pd:
        try:
            tmp = io.BytesIO(b)
            xls = pd.read_excel(tmp, sheet_name=None, engine="openpyxl")
            parts = []
            for sheet_name, df in xls.items():
                parts.append(f"--- Sheet: {sheet_name} ---")
                for _, row in df.fillna("").iterrows():
                    parts.append(" | ".join([str(x) for x in row.tolist()]))
            return _clean_text("\n".join(parts))
        except Exception:
            return ""
    return ""

def extract_text_from_csv_bytes(b: bytes) -> str:
    if pd:
        try:
            tmp = io.BytesIO(b)
            df = pd.read_csv(tmp)
            parts = []
            for _, row in df.fillna("").iterrows():
                parts.append(" | ".join([str(x) for x in row.tolist()]))
            return _clean_text("\n".join(parts))
        except Exception:
            return ""
    return ""

def extract_text_from_bytes(b: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf_bytes(b)
    elif ext == ".docx":
        return extract_text_from_docx_bytes(b)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_from_excel_bytes(b)
    elif ext == ".csv":
        return extract_text_from_csv_bytes(b)
    elif ext == ".txt":
        try:
            return _clean_text(b.decode("utf-8", errors="ignore"))
        except Exception:
            return ""
    else:
        return ""
