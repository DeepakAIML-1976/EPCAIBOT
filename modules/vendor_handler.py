# modules/vendor_handler.py
import streamlit as st
import os
import re
from modules import embedding_handler as eh
from modules import parser

VENDOR_DIR = "data/vendor_docs"
os.makedirs(VENDOR_DIR, exist_ok=True)

def guess_vendor_name(filename: str, text: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.split(r"[-_]", base, maxsplit=1)
    if m and m[0]:
        return m[0].strip()
    m2 = re.search(r"\b([A-Z][A-Z][A-Z][A-Z][A-Z]+(?:\s+[A-Z]+)*)\b", text[:500])
    return m2.group(1).title() if m2 else base

def save_and_index_vendor(uploaded_file, datasheet_target: str, model_choice="openai"):
    bytes_data = uploaded_file.read()
    text = parser.extract_text_from_bytes(bytes_data, uploaded_file.name)
    if text and text.strip():
        vendor_name = guess_vendor_name(uploaded_file.name, text)
        eh.index_document(
            text,
            source=uploaded_file.name,
            model_choice=model_choice,
            namespace="vendor_docs",
            extra_meta={
                "datasheet_id": datasheet_target,
                "vendor_doc": uploaded_file.name,
                "vendor_name": vendor_name
            }
        )
        eh.link_vendor_to_datasheet(datasheet_target, uploaded_file.name, vendor_name=vendor_name)
    with open(os.path.join(VENDOR_DIR, uploaded_file.name), "wb") as f:
        f.write(bytes_data)
    return text

def vendor_ui():
    st.header("📂 Vendor Document Upload & Link to Datasheet")
    model_choice = st.selectbox("Select embedding model", ["openai", "scibert", "matscibert"])
    datasheets = eh.get_datasheets()
    if not datasheets:
        st.info("Please upload datasheets first.")
        return
    datasheet_target = st.selectbox("Select datasheet to link vendor offers", datasheets)
    uploaded_files = st.file_uploader(
        "Upload Vendor Documents (PDF, DOCX, XLSX, CSV, TXT)",
        type=["pdf","docx","xlsx","xls","csv","txt"],
        accept_multiple_files=True
    )
    if uploaded_files and datasheet_target:
        for idx, f in enumerate(uploaded_files):
            with st.spinner(f"Processing {f.name} ..."):
                text = save_and_index_vendor(f, datasheet_target, model_choice=model_choice)
                if text:
                    st.success(f"Indexed vendor document: {f.name} (linked to {datasheet_target})")
                    st.text_area(
                        f"Preview: {f.name}",
                        text[:2000],
                        height=200,
                        key=f"vendor_preview_{idx}_{f.name}"
                    )
                else:
                    st.warning(f"Could not extract text from {f.name}.")
