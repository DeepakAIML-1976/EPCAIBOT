import os
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize SciBERT embeddings model
EMBED_MODEL = "allenai/scibert_scivocab_uncased"
model = SentenceTransformer(EMBED_MODEL)

# Embedding storage
EMBED_STORE_PATH = "data/embeddings.pkl"
if os.path.exists(EMBED_STORE_PATH):
    with open(EMBED_STORE_PATH, "rb") as f:
        embedding_store = pickle.load(f)
else:
    embedding_store = []

def add_embedding(text, metadata):
    """Add a new embedding with associated metadata"""
    embedding = model.encode(text, convert_to_tensor=True)
    embedding_store.append({"text": text, "embedding": embedding, "metadata": metadata})
    save_store()

def save_store():
    with open(EMBED_STORE_PATH, "wb") as f:
        pickle.dump(embedding_store, f)

def search_embedding(query, top_k=5):
    """Return top_k similar texts to the query"""
    if not embedding_store:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)
    results = []
    for item in embedding_store:
        score = util.pytorch_cos_sim(query_embedding, item["embedding"]).item()
        results.append({"text": item["text"], "metadata": item["metadata"], "score": score})

    # Sort descending by similarity
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
