# modules/training_handler.py
import os
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from modules import embedding_handler as eh

TRAIN_DIR = os.path.join("data", "training")
os.makedirs(TRAIN_DIR, exist_ok=True)
TRAIN_JSON = os.path.join(TRAIN_DIR, "training_data.json")
INDEX_FILE = os.path.join(TRAIN_DIR, "training.index")
META_FILE = os.path.join(TRAIN_DIR, "training_meta.json")

_state = {"index": None, "meta": []}

def _load_state():
    if os.path.exists(INDEX_FILE):
        try:
            _state["index"] = faiss.read_index(INDEX_FILE)
        except Exception:
            _state["index"] = None
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                _state["meta"] = json.load(f)
        except Exception:
            _state["meta"] = []

def _save_state():
    if _state["index"] is not None:
        faiss.write_index(_state["index"], INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(_state["meta"], f, indent=2)

def _ensure_index(model_choice: str = "openai"):
    if _state["index"] is None:
        dim = 1536 if model_choice == "openai" else 768
        _state["index"] = faiss.IndexFlatL2(dim)

def save_approved_tqs(approved_tqs: List[Dict[str, Any]], model_choice: str = "openai") -> int:
    if not approved_tqs:
        return 0
    _load_state()
    _ensure_index(model_choice=model_choice)
    added = 0
    vecs = []
    metas = []
    for item in approved_tqs:
        text = item.get("tq_text","").strip()
        if not text:
            continue
        emb = eh.get_embedding(text, model_choice=model_choice)
        vecs.append(emb.astype("float32"))
        metas.append({
            "datasheet": item.get("datasheet"),
            "vendor": item.get("vendor"),
            "vendor_doc": item.get("vendor_doc"),
            "tq_text": text,
            "ts": time.time()
        })
        added += 1
    if not vecs:
        return 0
    mat = np.vstack(vecs).astype("float32")
    _state["index"].add(mat)
    _state["meta"].extend(metas)
    _save_state()

    # append raw training data
    try:
        existing = []
        if os.path.exists(TRAIN_JSON):
            with open(TRAIN_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.extend(metas)
        with open(TRAIN_JSON, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass
    return added

def search_training(query: str, top_k: int = 5, model_choice: str = "openai") -> List[Dict[str, Any]]:
    _load_state()
    if _state["index"] is None or _state["index"].ntotal == 0:
        return []
    qv = eh.get_embedding(query, model_choice=model_choice).reshape(1,-1).astype("float32")
    D, I = _state["index"].search(qv, top_k)
    out = []
    for dist, i in zip(D[0], I[0]):
        if 0 <= i < len(_state["meta"]):
            meta = _state["meta"][i].copy()
            meta["score"] = float(dist)
            out.append(meta)
    return out

def get_training_stats():
    _load_state()
    n = _state["index"].ntotal if _state["index"] is not None else 0
    return {"examples": len(_state["meta"]), "index_size": int(n)}

def reset_training_store():
    _load_state()
    _state["index"] = None
    _state["meta"] = []
    for f in [TRAIN_JSON, INDEX_FILE, META_FILE]:
        if os.path.exists(f):
            try: os.remove(f)
            except Exception: pass
    os.makedirs(TRAIN_DIR, exist_ok=True)
