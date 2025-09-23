import streamlit as st

# Example user database
USER_DB = {
    "engineer1": "password123",
    "engineer2": "securepass"
}

def authenticate_user():
    """
    Simple Streamlit authentication for RFQ upload.
    Returns username if authenticated, else None.
    """
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['username'] = None

    if not st.session_state['authenticated']:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in USER_DB and USER_DB[username] == password:
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid credentials")
        return None
    else:
        return st.session_state['username']
