# modules/tq_generator.py
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from modules import embedding_handler as eh
from modules.email_handler import send_email
from modules import training_handler

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def _gather_context_for_pair(datasheet_name: str, vendor_doc: str, model_choice: str = "openai"):
    # fetch chunks for this datasheet
    ds_hits = eh.search(datasheet_name, top_k=8, model_choice=model_choice, namespace="datasheets")
    ds_hits = [h for h in ds_hits if h.get("source") == datasheet_name]
    # fetch chunks for this vendor doc linked to that datasheet
    v_hits = eh.search(vendor_doc, top_k=8, model_choice=model_choice, namespace="vendor_docs")
    v_hits = [h for h in v_hits if h.get("filename") == vendor_doc and h.get("datasheet_id") == datasheet_name]
    def short(chunks):
        return "\n".join([c["text"][:600].replace("\n", " ") for c in chunks[:6]])
    return short(ds_hits), short(v_hits)

from modules.tbe_generator import get_comparison_diff

def _draft_tqs(datasheet_name: str, vendor_doc: str, model_choice: str = "openai") -> str:
    ds_ctx, v_ctx = _gather_context_for_pair(datasheet_name, vendor_doc, model_choice=model_choice)
    
    # Get structured diffs
    diff_ctx = get_comparison_diff(datasheet_name, vendor_doc, model_choice=model_choice)

    prompt = f"""You are an EPC engineer drafting concise Technical Queries (TQs).
Compare the datasheet requirements to the specific vendor offer (do not include other vendors).
Refer to the 'DETECTED DEVIATIONS' below for specific discrepancies found in the data parameters.
Identify ambiguities, missing data, deviations, or clarifications needed.

Produce a numbered list of single-sentence TQs. Keep them precise and reference vendor by name if visible.

DATASHEET: {datasheet_name}
DETECTED DEVIATIONS (from TBE analysis):
{diff_ctx}

DATASHEET CONTEXT:
{ds_ctx}

VENDOR OFFER: {vendor_doc}
VENDOR CONTEXT:
{v_ctx}
"""

    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are a senior EPC engineer."},{"role":"user","content":prompt}],
                temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(Fallback) OpenAI error: {e}\n\n{prompt[:2000]}"
    return f"(Fallback) OpenAI not configured.\n\n{prompt[:2000]}"

def _extract_vendor_email_with_openai(text: str) -> str:
    if not client:
        return ""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"Extract possible email address for the vendor from the given text. Return only the email or empty."},
                      {"role":"user","content":text[:4000]}],
            temperature=0
        )
        email = (resp.choices[0].message.content or "").strip()
        return email if "@" in email else ""
    except Exception:
        return ""

def tq_ui():
    st.header("📌 Technical Query (TQ) Generator – Per Datasheet & Vendor")
    model_choice = st.selectbox("Select embedding model", ["openai", "scibert", "matscibert"])
    datasheets = eh.get_datasheets()
    if not datasheets:
        st.info("Please upload datasheets and vendor offers first.")
        return
    ds = st.selectbox("Choose datasheet", datasheets)
    vendors = eh.get_vendors_for_datasheet(ds)
    if not vendors:
        st.info("No vendor offers linked to this datasheet yet.")
        return
    vendor_doc = st.selectbox("Choose vendor offer", vendors)
    vendor_display = eh.get_vendor_display_name(vendor_doc)

    if st.button("Generate TQs for this pair"):
        with st.spinner("Generating TQs..."):
            tqs = _draft_tqs(ds, vendor_doc, model_choice=model_choice)
            st.session_state.setdefault("tq_generated", {})
            st.session_state["tq_generated"][(ds, vendor_doc)] = tqs

    # Show generated TQs with approval checkboxes
    tqs_text = st.session_state.get("tq_generated", {}).get((ds, vendor_doc), "")
    if tqs_text:
        st.subheader(f"Draft TQs — {vendor_display} vs {ds}")
        lines = [ln.strip() for ln in tqs_text.split("\n") if ln.strip()]
        approved_key_prefix = f"approve_{ds}_{vendor_doc}_"
        approved_list = []
        for i, ln in enumerate(lines, 1):
            checked = st.checkbox(ln, key=approved_key_prefix+str(i))
            if checked:
                approved_list.append(ln)

        # Save approved TQs to session (for Show All / Email / Training)
        if st.button("Save approvals for this pair"):
            st.session_state.setdefault("approved_tqs", [])
            for ln in approved_list:
                st.session_state["approved_tqs"].append({
                    "datasheet": ds,
                    "vendor": vendor_display,
                    "vendor_doc": vendor_doc,
                    "tq_text": ln
                })
            st.success(f"Saved {len(approved_list)} approved TQs for {vendor_display} / {ds}")

    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        if st.button("Show all approved TQs"):
            all_ = st.session_state.get("approved_tqs", [])
            if not all_:
                st.info("No approved TQs yet.")
            else:
                st.subheader("✅ Approved TQs")
                grouped = {}
                for item in all_:
                    grouped.setdefault((item["datasheet"], item["vendor"]), []).append(item["tq_text"])
                for (dsn, vname), lst in grouped.items():
                    st.markdown(f"**{dsn} — {vname}**")
                    st.write("\n".join([f"- {x}" for x in lst]))
                    st.markdown("---")
    with cols[1]:
        if st.button("Draft email to vendor"):
            all_ = st.session_state.get("approved_tqs", [])
            sel = [x for x in all_ if x["datasheet"] == ds and x["vendor_doc"] == vendor_doc]
            if not sel:
                st.warning("No approved TQs for the selected pair.")
            else:
                # try to get vendor email from index text (best-effort)
                vendor_text_hits = eh.search(vendor_doc, top_k=5, model_choice=model_choice, namespace="vendor_docs")
                concatenated = "\n".join([h["text"] for h in vendor_text_hits])
                email_guess = _extract_vendor_email_with_openai(concatenated) if concatenated else ""

                st.session_state["draft_subject"] = f"Technical Queries regarding {ds}"
                st.session_state["draft_to"] = email_guess
                body = f"Dear {eh.get_vendor_display_name(vendor_doc)},\n\nPlease find below our technical queries regarding your submission for {ds}:\n"
                for i, item in enumerate(sel, 1):
                    body += f"{i}. {item['tq_text']}\n"
                body += "\nKind regards,\nEPC Engineering Team"
                st.session_state["draft_body"] = body
                st.success("Draft prepared below:")

    with cols[2]:
        if st.button("Save approved TQs to Learned Memory"):
            all_ = st.session_state.get("approved_tqs", [])
            saved = training_handler.save_approved_tqs(all_, model_choice=model_choice)
            st.success(f"Saved {saved} TQ(s) to the learned memory.")

    # Draft form (if created)
    if st.session_state.get("draft_body"):
        st.subheader("📧 Email Draft")
        to_addr = st.text_input("To (vendor email)", value=st.session_state.get("draft_to",""))
        subject = st.text_input("Subject", value=st.session_state.get("draft_subject",""))
        body = st.text_area("Body", value=st.session_state.get("draft_body",""), height=300)
        if st.button("Send Email"):
            if not to_addr or "@" not in to_addr:
                st.warning("Please enter a valid recipient email or leave blank if unavailable.")
            else:
                ok = send_email(subject, body, to_addr)
                if ok:
                    st.success(f"Email sent to {to_addr}")
