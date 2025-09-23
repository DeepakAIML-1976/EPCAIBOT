# modules/embedding_handler.py
import os
import time
import threading
import pickle
import shutil
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DATA_DIR = "data"
FAISS_DIR = os.path.join(DATA_DIR, "faiss_indexes")
REGISTRY_FILE = os.path.join(DATA_DIR, "registry.json")  # datasheets & vendor link map
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "openai": {"dim": 1536, "hf_name": None},
    "scibert": {"dim": 768, "hf_name": "allenai/scibert_scivocab_uncased"},
    "matscibert": {"dim": 768, "hf_name": "m3rg-iitd/matscibert"},
}

_indexes_lock = threading.Lock()
indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}
_st_models: Dict[str, SentenceTransformer] = {}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI is not None) else None

def _model_dim(model_name: str) -> int:
    return MODEL_CONFIGS[model_name]["dim"]

def _hf_model_name(model_name: str) -> Optional[str]:
    return MODEL_CONFIGS[model_name]["hf_name"]

def _ensure_namespace_index(model_name: str, namespace: str = "general"):
    with _indexes_lock:
        if model_name not in indexes:
            indexes[model_name] = {}
        if namespace in indexes[model_name]:
            return indexes[model_name][namespace]
        dim = _model_dim(model_name)
        idx = faiss.IndexFlatL2(dim)
        entry = {"index": idx, "docs": []}
        indexes[model_name][namespace] = entry
        _load_index_from_disk(model_name, namespace, entry)
        return entry

def _hf_sentence_transformer(model_name: str) -> SentenceTransformer:
    hf = _hf_model_name(model_name)
    if hf is None:
        raise ValueError(f"No HF model configured for {model_name}")
    if hf not in _st_models:
        _st_models[hf] = SentenceTransformer(hf)
    return _st_models[hf]

def get_embedding(text: str, model_choice: str = "openai") -> np.ndarray:
    model_choice = model_choice.lower()
    if model_choice not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_choice: {model_choice}")
    if model_choice == "openai":
        if openai_client is None:
            raise RuntimeError("OpenAI client not configured (OPENAI_API_KEY missing).")
        resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding, dtype="float32")
    st_model = _hf_sentence_transformer(model_choice)
    vec = st_model.encode(text, convert_to_numpy=True)
    return np.array(vec, dtype="float32")

def _index_file_path(model_name: str, namespace: str) -> Tuple[str, str]:
    dir_path = os.path.join(FAISS_DIR, model_name)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"{namespace}.index"), os.path.join(dir_path, f"{namespace}.pkl")

def _save_index_to_disk(model_name: str, namespace: str, entry: Dict[str, Any]):
    idx_path, meta_path = _index_file_path(model_name, namespace)
    try:
        faiss.write_index(entry["index"], idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(entry["docs"], f)
    except Exception as e:
        print(f"[embedding_handler] persist error {model_name}/{namespace}: {e}")

def _load_index_from_disk(model_name: str, namespace: str, entry: Dict[str, Any]):
    idx_path, meta_path = _index_file_path(model_name, namespace)
    try:
        if os.path.exists(idx_path):
            entry["index"] = faiss.read_index(idx_path)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                entry["docs"] = pickle.load(f)
    except Exception as e:
        print(f"[embedding_handler] load error {model_name}/{namespace}: {e}")

def persist_all_indexes():
    with _indexes_lock:
        for model_name, ns_map in indexes.items():
            for namespace, entry in ns_map.items():
                _save_index_to_disk(model_name, namespace, entry)

def text_to_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > 0 else end
    return chunks

def index_document(
    text: str,
    source: str,
    model_choice: str = "openai",
    namespace: Optional[str] = "general",
    extra_meta: Optional[Dict[str, Any]] = None,
    chunk_size: int = 800,
    overlap: int = 120,
) -> int:
    model_choice = model_choice.lower()
    ns_entry = _ensure_namespace_index(model_choice, namespace)
    chunks = text_to_chunks(text, chunk_size=chunk_size, overlap=overlap)
    vectors, metas = [], []
    for i, c in enumerate(chunks):
        try:
            emb = get_embedding(c, model_choice=model_choice)
        except Exception as e:
            print(f"[embedding_handler] embedding error {source} chunk {i}: {e}")
            continue
        vectors.append(emb)
        meta = {
            "source": source,
            "namespace": namespace,
            "chunk_id": i,
            "text": c,
            "filename": source,
            "created_ts": time.time()
        }
        if extra_meta:
            meta.update(extra_meta)
        metas.append(meta)
    if not vectors:
        return 0
    mat = np.vstack(vectors).astype("float32")
    with _indexes_lock:
        ns_entry["index"].add(mat)
        ns_entry["docs"].extend(metas)
    _save_index_to_disk(model_choice, namespace, ns_entry)
    return len(vectors)

def search(query: str, top_k: int = 5, model_choice: str = "openai", namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    model_choice = model_choice.lower()
    qv = get_embedding(query, model_choice=model_choice).reshape(1, -1).astype("float32")
    results: List[Dict[str, Any]] = []
    with _indexes_lock:
        model_indexes = indexes.get(model_choice, {})
        namespaces = [namespace] if namespace else list(model_indexes.keys())
        for ns in namespaces:
            ns_entry = model_indexes.get(ns)
            if not ns_entry or ns_entry["index"].ntotal == 0:
                continue
            D, I = ns_entry["index"].search(qv, top_k)
            for dist, i in zip(D[0], I[0]):
                if i < len(ns_entry["docs"]):
                    meta = ns_entry["docs"][i]
                    results.append({**meta, "score": float(dist)})
    results.sort(key=lambda x: x["score"])
    return results

def get_index_stats(model_choice: str = "openai") -> Dict[str, Any]:
    stats = {}
    with _indexes_lock:
        mdl = indexes.get(model_choice, {})
        for ns, entry in mdl.items():
            stats[ns] = {"ntotal": int(entry["index"].ntotal), "docs": len(entry["docs"])}
    return stats

# ---- Registry helpers (datasheet ↔ vendor link map) ----
import json
def _load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_registry(reg: Dict[str, Any]):
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
    except Exception as e:
        print(f"[embedding_handler] registry save error: {e}")

def add_datasheet_entry(datasheet_name: str):
    reg = _load_registry()
    reg.setdefault("datasheets", {})
    reg["datasheets"].setdefault(datasheet_name, {"vendors": []})
    _save_registry(reg)

def link_vendor_to_datasheet(datasheet_name: str, vendor_doc_name: str, vendor_name: Optional[str] = None):
    reg = _load_registry()
    reg.setdefault("datasheets", {})
    ds = reg["datasheets"].setdefault(datasheet_name, {"vendors": []})
    if vendor_doc_name not in ds["vendors"]:
        ds["vendors"].append(vendor_doc_name)
    # track vendor names
    reg.setdefault("vendors", {})
    if vendor_name:
        reg["vendors"][vendor_doc_name] = {"name": vendor_name}
    _save_registry(reg)

def get_datasheets() -> List[str]:
    reg = _load_registry()
    return sorted(list(reg.get("datasheets", {}).keys()))

def get_vendors_for_datasheet(datasheet_name: str) -> List[str]:
    reg = _load_registry()
    return reg.get("datasheets", {}).get(datasheet_name, {}).get("vendors", [])

def get_vendor_display_name(vendor_doc_name: str) -> str:
    reg = _load_registry()
    return reg.get("vendors", {}).get(vendor_doc_name, {}).get("name", vendor_doc_name)

# ---- Reset all data ----
def reset_all_data():
    with _indexes_lock:
        indexes.clear()
    # delete data subdirs
    for sub in ["datasheets", "rfqs", "vendor_docs", "faiss_indexes", "training"]:
        p = os.path.join(DATA_DIR, sub)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
    # delete registry and misc
    for f in ["registry.json", "embeddings.pkl", "training_data.json", "training_index.pkl"]:
        fp = os.path.join(DATA_DIR, f)
        if os.path.exists(fp):
            try: os.remove(fp)
            except Exception: pass
    os.makedirs(FAISS_DIR, exist_ok=True)
