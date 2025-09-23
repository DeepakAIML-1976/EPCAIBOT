# modules/embedding_handler.py

import os
import json
from pathlib import Path
import numpy as np
import faiss
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "indexes"
META_DIR = DATA_DIR / "meta"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

MODEL_CONFIG = {
    "openai": {"dim": 1536},
    "scibert": {"dim": 768, "hf": "allenai/scibert_scivocab_uncased"},
    "matscibert": {"dim": 768, "hf": "m3rg-iitd/matscibert"},
}

_hf_cache = {}

def _hf_model(name):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    if name not in _hf_cache:
        _hf_cache[name] = SentenceTransformer(name)
    return _hf_cache[name]

def _openai_embed(text, model="text-embedding-3-small"):
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = openai_client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def embed_text(text, model_choice="openai"):
    model_choice = model_choice.lower()
    if model_choice == "openai":
        return _openai_embed(text)
    cfg = MODEL_CONFIG.get(model_choice)
    if not cfg or "hf" not in cfg:
        raise ValueError("Unknown model choice or HF model not configured")
    hf = _hf_model(cfg["hf"])
    vec = hf.encode(text, convert_to_numpy=True)
    return np.array(vec, dtype="float32")

# simple per-namespace index management
def _index_paths(model_choice, namespace):
    safe = f"{model_choice}__{namespace}"
    return INDEX_DIR / f"{safe}.index", META_DIR / f"{safe}.meta.json"

def load_index(model_choice, namespace):
    idx_path, meta_path = _index_paths(model_choice, namespace)
    if not idx_path.exists():
        dim = MODEL_CONFIG.get(model_choice, {}).get("dim", 1536)
        return faiss.IndexFlatL2(dim), []
    idx = faiss.read_index(str(idx_path))
    meta = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return idx, meta

def save_index(index, meta, model_choice, namespace):
    idx_path, meta_path = _index_paths(model_choice, namespace)
    faiss.write_index(index, str(idx_path))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

def index_document(text, source="unknown", model_choice="openai", namespace="default"):
    vec = embed_text(text, model_choice=model_choice)
    idx, meta = load_index(model_choice, namespace)
    if idx.ntotal == 0:
        # ensure index dimension matches
        idx = faiss.IndexFlatL2(len(vec))
    idx.add(np.array([vec]).astype("float32"))
    meta.append({"source": source, "text": text})
    save_index(idx, meta, model_choice, namespace)

def search(query, top_k=5, model_choice="openai", namespace=None):
    vec = embed_text(query, model_choice=model_choice)
    # if namespace specified, search that only
    if namespace:
        idx, meta = load_index(model_choice, namespace)
        if idx.ntotal == 0:
            return []
        D, I = idx.search(np.array([vec]).astype("float32"), top_k)
        results = []
        for i in I[0]:
            if i < len(meta):
                results.append(meta[i])
        return results
    # otherwise search all index files for this model
    results = []
    for p in INDEX_DIR.glob(f"{model_choice}__*.index"):
        name = p.stem
        ns = name.split("__", 1)[1]
        idx, meta = load_index(model_choice, ns)
        if idx.ntotal == 0:
            continue
        D, I = idx.search(np.array([vec]).astype("float32"), top_k)
        for i in I[0]:
            if i < len(meta):
                results.append(meta[i])
    return results

def persist_all_indexes():
    # indexes are saved immediately on index_document; this is a stub to keep API stable
    return True

def load_all_indexes():
    loaded = []
    for p in INDEX_DIR.glob("*.index"):
        # attempt to read to check validity
        try:
            faiss.read_index(str(p))
            loaded.append(p.name)
        except Exception:
            pass
    return loaded

def get_index_stats(model_choice="openai"):
    stats = []
    for p in INDEX_DIR.glob(f"{model_choice}__*.index"):
        try:
            idx = faiss.read_index(str(p))
            stats.append({"file": p.name, "ntotal": idx.ntotal})
        except Exception:
            stats.append({"file": p.name, "ntotal": None})
    return stats

def reset_all_data():
    # remove index files and meta
    for f in (INDEX_DIR.glob("*") if INDEX_DIR.exists() else []):
        try:
            f.unlink()
        except Exception:
            pass
    for f in (META_DIR.glob("*") if META_DIR.exists() else []):
        try:
            f.unlink()
        except Exception:
            pass
    return True
