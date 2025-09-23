# utils/file_utils.py

from typing import Union
import io
import re

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None

try:
    import pandas as pd
except Exception:
    pd = None

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def _extract_pdf_bytes(b: bytes) -> str:
    parts = []
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        parts.append(t)
            res = "\n".join(parts)
            if res.strip():
                return _clean_text(res)
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
                    parts.append(t)
            return _clean_text("\n".join(parts))
        except Exception:
            pass
    try:
        return _clean_text(b.decode("utf-8", errors="ignore"))
    except Exception:
        return ""

def _extract_docx_bytes(b: bytes) -> str:
    if docx:
        try:
            tmp = io.BytesIO(b)
            document = docx.Document(tmp)
            paras = [p.text for p in document.paragraphs if p.text]
            return _clean_text("\n".join(paras))
        except Exception:
            return ""
    return ""

def _extract_xlsx_bytes(b: bytes) -> str:
    if pd:
        try:
            tmp = io.BytesIO(b)
            xls = pd.read_excel(tmp, sheet_name=None, engine="openpyxl")
            parts = []
            for name, df in xls.items():
                parts.append(f"--- Sheet: {name} ---")
                for _, row in df.fillna("").iterrows():
                    parts.append(" | ".join([str(x) for x in row.tolist()]))
            return _clean_text("\n".join(parts))
        except Exception:
            return ""
    return ""

def _extract_csv_bytes(b: bytes) -> str:
    if pd:
        try:
            tmp = io.BytesIO(b)
            df = pd.read_csv(tmp)
            parts = []
            for _, row in df.fillna("").iterrows():
                parts.append(" | ".join([str(x) for x in row.tolist()]))
            return _clean_text("\n".join(parts))
        except Exception:
            try:
                return _clean_text(b.decode("utf-8", errors="ignore"))
            except Exception:
                return ""
    else:
        try:
            return _clean_text(b.decode("utf-8", errors="ignore"))
        except Exception:
            return ""

def extract_text(uploaded_file: Union[io.BytesIO, "streamlit.runtime.uploaded_file_manager.UploadedFile"]) -> str:
    try:
        name = getattr(uploaded_file, "name", None)
        uploaded_file.seek(0)
        b = uploaded_file.read()
    except Exception:
        if isinstance(uploaded_file, (bytes, bytearray)):
            b = bytes(uploaded_file)
            name = None
        else:
            return ""

    ext = ""
    if name:
        ext = (name or "").lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return _extract_pdf_bytes(b)
    if ext == "docx":
        return _extract_docx_bytes(b)
    if ext in ("xls", "xlsx"):
        return _extract_xlsx_bytes(b)
    if ext == "csv":
        return _extract_csv_bytes(b)
    if ext == "txt" or ext == "":
        try:
            return _clean_text(b.decode("utf-8", errors="ignore"))
        except Exception:
            return ""
    # fallback
    txt = _extract_pdf_bytes(b)
    if txt:
        return txt
    try:
        return _clean_text(b.decode("utf-8", errors="ignore"))
    except Exception:
        return ""
