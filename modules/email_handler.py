# modules/email_handler.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER") or os.getenv("EMAIL_SENDER")
SMTP_PASS = os.getenv("SMTP_PASS") or os.getenv("EMAIL_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM") or SMTP_USER

def send_email(subject: str, body: str, to_email: str) -> bool:
    if not all([SMTP_USER, SMTP_PASS, EMAIL_FROM]):
        st.error("Email configuration missing. Set SMTP_USER, SMTP_PASS, EMAIL_FROM in .env")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_FROM
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(EMAIL_FROM, [to_email], msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email send failed: {e}")
        return False
