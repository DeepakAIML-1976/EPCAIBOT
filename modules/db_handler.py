# modules/db_handler.py

import sqlite3
import os

DB_PATH = "epc_ai.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialize the SQLite database with required tables if they don’t exist"""
    conn = get_connection()
    cur = conn.cursor()

    # Datasheets
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasheets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            content TEXT
        )
    """)

    # Vendors
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vendors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            datasheet_id INTEGER,
            content TEXT,
            FOREIGN KEY(datasheet_id) REFERENCES datasheets(id)
        )
    """)

    # TQs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datasheet_id INTEGER,
            vendor_id INTEGER,
            content TEXT,
            approved INTEGER DEFAULT 0,
            FOREIGN KEY(datasheet_id) REFERENCES datasheets(id),
            FOREIGN KEY(vendor_id) REFERENCES vendors(id)
        )
    """)

    # TBEs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tbes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datasheet_id INTEGER,
            datasheet_name TEXT,
            html_content TEXT,
            FOREIGN KEY(datasheet_id) REFERENCES datasheets(id)
        )
    """)

    # Emails
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT,
            content TEXT
        )
    """)

    conn.commit()
    conn.close()


# =============================
# Datasheets
# =============================

def save_datasheet(name, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO datasheets (name, content) VALUES (?, ?)", (name, content))
    conn.commit()
    ds_id = cur.lastrowid
    conn.close()
    return ds_id

def get_all_datasheets():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM datasheets")
    rows = cur.fetchall()
    conn.close()
    return rows


# =============================
# Vendors
# =============================

def save_vendor(name, datasheet_id, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO vendors (name, datasheet_id, content) VALUES (?, ?, ?)",
                (name, datasheet_id, content))
    conn.commit()
    v_id = cur.lastrowid
    conn.close()
    return v_id

def get_all_vendors():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vendors")
    rows = cur.fetchall()
    conn.close()
    return rows


# =============================
# TQs
# =============================

def save_tq(datasheet_id, vendor_id, content, approved=False):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO tqs (datasheet_id, vendor_id, content, approved) VALUES (?, ?, ?, ?)",
                (datasheet_id, vendor_id, content, int(approved)))
    conn.commit()
    tq_id = cur.lastrowid
    conn.close()
    return tq_id

def approve_tq(tq_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE tqs SET approved = 1 WHERE id = ?", (tq_id,))
    conn.commit()
    conn.close()

def get_all_tqs(only_approved=False):
    conn = get_connection()
    cur = conn.cursor()
    if only_approved:
        cur.execute("SELECT * FROM tqs WHERE approved = 1")
    else:
        cur.execute("SELECT * FROM tqs")
    rows = cur.fetchall()
    conn.close()
    return rows


# =============================
# TBEs
# =============================

def save_tbe(datasheet_id, datasheet_name, html_content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO tbes (datasheet_id, datasheet_name, html_content) VALUES (?, ?, ?)",
                (datasheet_id, datasheet_name, html_content))
    conn.commit()
    tbe_id = cur.lastrowid
    conn.close()
    return tbe_id

def get_all_tbes():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tbes")
    rows = cur.fetchall()
    conn.close()
    return rows


# =============================
# Emails
# =============================

def save_email(vendor_name, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO emails (vendor_name, content) VALUES (?, ?)", (vendor_name, content))
    conn.commit()
    email_id = cur.lastrowid
    conn.close()
    return email_id

def get_all_emails():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM emails")
    rows = cur.fetchall()
    conn.close()
    return rows
