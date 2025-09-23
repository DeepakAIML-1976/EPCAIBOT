import streamlit as st
import pandas as pd

def app():
    st.header("Compliance Matrix & TBE Report Generator")

    comparison_file = st.file_uploader("Upload Comparison File (CSV from Comparison Page)", type=["csv"])
    
    if comparison_file:
        df = pd.read_csv(comparison_file)
        
        # Generate Compliance Matrix
        st.subheader("Compliance Matrix")
        compliance_cols = [col for col in df.columns if col.endswith("_MATCH")]
        df['COMPLIANT'] = df[compliance_cols].all(axis=1)
        st.dataframe(df[['EQUIPMENT_NAME', 'COMPLIANT'] + compliance_cols])

        # Generate TBE Report with source links
        st.subheader("TBE Report")
        df['TBE_LINK'] = df.apply(lambda x: f"https://internal_system.com/docs/{x['EQUIPMENT_NAME']}", axis=1)
        st.dataframe(df[['EQUIPMENT_NAME', 'TBE_LINK', 'COMPLIANT'] + compliance_cols])

        st.success("Compliance Matrix and TBE Report generated successfully!")
