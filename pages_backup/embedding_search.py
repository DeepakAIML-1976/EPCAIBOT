import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from modules.tq_generator import generate_tq
import os
import pickle

# Load or initialize embeddings database
EMBEDDINGS_FILE = "data/embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight, fast

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

def app():
    st.header("Live Embedding Search & One-Click TQ Generation")

    model = load_model()
    embeddings_db = load_embeddings()

    uploaded_files = st.file_uploader("Upload Historical RFQs/Datasheets (for search index)", accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            text = ""
            if file.name.endswith(".pdf"):
                from modules.parser import parse_pdf
                text = parse_pdf(file)
            elif file.name.endswith(".xlsx"):
                from modules.parser import parse_excel
                df = parse_excel(file)
                text = " ".join(df.astype(str).values.flatten())
            embedding = model.encode(text)
            embeddings_db[file.name] = {"text": text, "embedding": embedding}
        save_embeddings(embeddings_db)
        st.success("Embeddings updated!")

    query = st.text_input("Enter query to search historical references")
    if query and embeddings_db:
        query_embedding = model.encode(query)
        scores = {}
        for fname, data in embeddings_db.items():
            sim_score = util.cos_sim(query_embedding, data['embedding']).item()
            scores[fname] = sim_score
        # Top 3 references
        top_refs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        st.subheader("Top Historical References")
        for fname, score in top_refs:
            st.markdown(f"**{fname}** (Similarity: {score:.2f})")
            st.text_area(f"Reference: {fname}", value=embeddings_db[fname]['text'], height=150)
            if st.button(f"Generate TQ from {fname}"):
                tq = generate_tq(embeddings_db[fname]['text'])
                st.text_area(f"TQ from {fname}", value=tq, height=150)
