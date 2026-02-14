import streamlit as st
import pandas as pd
from modules.normalizer import normalize_attributes

def app():
    st.header("Automated Requirement vs. Offer Comparison")

    req_file = st.file_uploader("Upload Requirement Datasheet (CSV)", type=["csv"], key="req")
    offer_file = st.file_uploader("Upload Vendor Offer (CSV)", type=["csv"], key="offer")

    if req_file and offer_file:
        df_req = pd.read_csv(req_file)
        df_offer = pd.read_csv(offer_file)

        # Normalize to ontology
        df_req_norm = normalize_attributes(df_req)
        df_offer_norm = normalize_attributes(df_offer)

        st.subheader("Comparison Results")
        merged = pd.merge(df_req_norm, df_offer_norm, on="EQUIPMENT_NAME", how="outer", suffixes=("_REQ", "_OFFER"))
        
        # Flag mismatches
        for col in ["DISCIPLINE", "MATERIAL", "SPECIFICATION"]:
            merged[f"{col}_MATCH"] = merged[f"{col}_REQ"] == merged[f"{col}_OFFER"]

        st.dataframe(merged)
        st.success("Comparison completed! Columns with *_MATCH indicate alignment between requirement and offer.")
