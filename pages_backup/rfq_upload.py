import streamlit as st
from utils.auth import authenticate_user
from utils.storage import save_rfq_file
from modules.parser import parse_pdf, parse_excel

def app():
    st.header("RFQ Upload Portal (Secure & Access-Controlled)")

    username = authenticate_user()
    if not username:
        st.info("Please login to upload RFQs.")
        return

    st.subheader("Upload RFQ File")
    uploaded_file = st.file_uploader("Choose RFQ (PDF or Excel)", type=["pdf", "xlsx"])
    
    if uploaded_file:
        file_path = save_rfq_file(uploaded_file, uploaded_file.name)
        st.success(f"RFQ uploaded successfully by {username}!")
        
        # Auto-parse uploaded RFQ
        if uploaded_file.name.endswith(".pdf"):
            text = parse_pdf(file_path)
            st.text_area("Parsed RFQ Text", value=text, height=300)
        elif uploaded_file.name.endswith(".xlsx"):
            df = parse_excel(file_path)
            st.dataframe(df)
