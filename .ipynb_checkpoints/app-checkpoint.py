# app.py
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

from modules import (
    datasheet_handler,
    rfq_handler,
    vendor_handler,
    tq_generator,
    tbe_generator,
    embedding_handler,
    training_handler
)

st.set_page_config(page_title="EPC AI BOT", page_icon="⚙️", layout="wide")
st.title("⚙️ EPC AI BOT – Datasheet → Offer → TQ/TBE Automation")

# ---- Sidebar ----
with st.sidebar:
    st.title("📂 Navigation")
    page = st.radio(
        "Go to:",
        [
            "Datasheet Upload",
            "Vendor Submissions",
            "RFQ Upload",
            "Embedding Search",
            "TQ Generator",
            "TBE Generator",
            "Admin / Reset",
        ],
    )
    st.markdown("---")
    st.subheader("🧠 Learned Memory")
    if st.button("Show Training Stats"):
        st.json(training_handler.get_training_stats())

# ---- Pages ----
if page == "Datasheet Upload":
    datasheet_handler.datasheet_ui()

elif page == "Vendor Submissions":
    vendor_handler.vendor_ui()

elif page == "RFQ Upload":
    rfq_handler.rfq_ui()

elif page == "Embedding Search":
    st.header("🔍 Embedding Search")
    model_choice = st.selectbox("Select embedding model:", ["openai", "scibert", "matscibert"])
    query = st.text_input("Enter search query:")
    ns = st.selectbox("Namespace (optional)", ["", "datasheets", "rfqs", "vendor_docs", "training"])
    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                namespace = ns if ns else None
                # search across main store
                results = embedding_handler.search(query, top_k=5, model_choice=model_choice, namespace=namespace)
                # search learned memory too
                learned = training_handler.search_training(query, top_k=3, model_choice=model_choice)
                if not results and not learned:
                    st.info("No results found.")
                else:
                    st.subheader("Document Index Results")
                    for r in results:
                        st.markdown(f"**Source:** {r['source']} — **Namespace:** {r.get('namespace','')} — **chunk:** {r.get('chunk_id')}")
                        st.write(r["text"][:1500])
                        st.markdown("---")
                    st.subheader("Learned Memory Results (Approved TQs)")
                    for r in learned:
                        st.markdown(f"**Vendor:** {r.get('vendor','?')} — **Datasheet:** {r.get('datasheet','?')} — **score:** {r.get('score'):.4f}")
                        st.write(r["tq_text"])
                        st.markdown("---")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Show Index Stats"):
            stats = embedding_handler.get_index_stats(model_choice=model_choice)
            st.json(stats)
    with cols[1]:
        if st.button("Persist all indexes to disk"):
            embedding_handler.persist_all_indexes()
            st.success("Persisted indexes to disk.")

elif page == "TQ Generator":
    tq_generator.tq_ui()

elif page == "TBE Generator":
    tbe_generator.tbe_ui()

elif page == "Admin / Reset":
    st.header("🛠️ Admin / Reset Workspace")
    st.warning("This will delete all uploaded files, FAISS indexes, and learned training data. This cannot be undone.")
    confirm = st.checkbox("I understand the consequences.")
    if st.button("Reset Workspace", type="primary") and confirm:
        embedding_handler.reset_all_data()
        training_handler.reset_training_store()
        st.success("Workspace reset complete.")
