# modules/rfq_handler.py
import streamlit as st
import os
from modules import embedding_handler as eh
from modules import parser

RFQ_DIR = "data/rfqs"
os.makedirs(RFQ_DIR, exist_ok=True)

def save_and_index_rfq(uploaded_file, model_choice="openai"):
    bytes_data = uploaded_file.read()
    text = parser.extract_text_from_bytes(bytes_data, uploaded_file.name)
    if text and text.strip():
        eh.index_document(text, source=uploaded_file.name, model_choice=model_choice, namespace="rfqs")
    with open(os.path.join(RFQ_DIR, uploaded_file.name), "wb") as f:
        f.write(bytes_data)
    return text

def rfq_ui():
    st.header("📄 RFQ Upload & Indexing")
    model_choice = st.selectbox("Select embedding model", ["openai", "scibert", "matscibert"])
    uploaded_files = st.file_uploader(
        "Upload RFQs (PDF, DOCX, XLSX, CSV, TXT)",
        type=["pdf","docx","xlsx","xls","csv","txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for idx, f in enumerate(uploaded_files):
            with st.spinner(f"Processing {f.name} ..."):
                text = save_and_index_rfq(f, model_choice=model_choice)
                if text:
                    st.success(f"Indexed RFQ: {f.name}")
                    st.text_area(
                        f"Preview: {f.name}",
                        text[:2000],
                        height=200,
                        key=f"rfq_preview_{idx}_{f.name}"
                    )
                else:
                    st.warning(f"Could not extract text from {f.name}.")
