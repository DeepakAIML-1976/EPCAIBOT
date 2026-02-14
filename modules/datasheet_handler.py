# modules/datasheet_handler.py
import streamlit as st
import os
from modules import embedding_handler as eh
from modules import parser

# Robust absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "datasheets")
os.makedirs(DATA_DIR, exist_ok=True)

def save_and_index_file(uploaded_file, model_choice="openai"):
    bytes_data = uploaded_file.read()
    text = parser.extract_text_from_bytes(bytes_data, uploaded_file.name)
    if text and text.strip():
        eh.index_document(
            text,
            source=uploaded_file.name,
            model_choice=model_choice,
            namespace="datasheets",
            extra_meta={"datasheet_id": uploaded_file.name}
        )
        eh.add_datasheet_entry(uploaded_file.name)
    
    # Ensure dir exists and use sanitized filename
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = os.path.basename(uploaded_file.name)
    with open(os.path.join(DATA_DIR, safe_name), "wb") as f:
        f.write(bytes_data)
    return text

def datasheet_ui():
    st.header("📑 Datasheet Upload & Indexing")
    model_choice = st.selectbox("Select embedding model", ["openai", "scibert", "matscibert"])
    uploaded_files = st.file_uploader(
        "Upload Datasheets (PDF, DOCX, XLSX, CSV, TXT)",
        type=["pdf","docx","xlsx","xls","csv","txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for idx, f in enumerate(uploaded_files):
            with st.spinner(f"Processing {f.name} ..."):
                text = save_and_index_file(f, model_choice=model_choice)
                if text:
                    st.success(f"Indexed datasheet: {f.name}")
                    st.text_area(
                        f"Preview: {f.name}",
                        text[:2000],
                        height=200,
                        key=f"ds_preview_{idx}_{f.name}"
                    )
                else:
                    st.warning(f"Could not extract text from {f.name}.")
    st.markdown("---")
    st.subheader("📋 Current Datasheets")
    st.write(eh.get_datasheets())
