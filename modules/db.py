# modules/db.py
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any

DB_PATH = os.path.join("data", "epc_ai.db")
os.makedirs("data", exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # documents: text blobs (could be truncated if you want), faiss_pos aligns with index insertion order
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        namespace TEXT,
        model TEXT,
        text TEXT,
        faiss_pos INTEGER,
        created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tqs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        vendor_email TEXT,
        status TEXT,
        created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tbes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        recipient_email TEXT,
        status TEXT,
        created_at TEXT
    );
    """)
    conn.commit()
    conn.close()

def insert_document(source: str, namespace: str, model: str, text: str, faiss_pos: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO documents (source, namespace, model, text, faiss_pos, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (source, namespace, model, text, faiss_pos, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_documents(model: str = None, namespace: str = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    q = "SELECT * FROM documents"
    params = []
    clauses = []
    if model:
        clauses.append("model = ?")
        params.append(model)
    if namespace:
        clauses.append("namespace = ?")
        params.append(namespace)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY faiss_pos ASC"
    cur.execute(q, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_tq(text: str, vendor_email: str, status: str = "approved"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tqs (text, vendor_email, status, created_at) VALUES (?, ?, ?, ?)
    """, (text, vendor_email, status, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_tqs() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tqs ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_tbe(text: str, recipient_email: str, status: str = "approved"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tbes (text, recipient_email, status, created_at) VALUES (?, ?, ?, ?)
    """, (text, recipient_email, status, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_tbes() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tbes ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
