import streamlit as st
import pandas as pd
from utils.storage import save_datasheet

def app():
    st.header("Online Datasheet Preparation")

    st.subheader("Enter Datasheet Details")
    
    # Dynamic form input for multiple entries
    equipment_list = []
    num_rows = st.number_input("Number of Equipment Entries", min_value=1, max_value=50, value=5)

    for i in range(num_rows):
        st.markdown(f"**Equipment {i+1}**")
        name = st.text_input(f"Equipment Name {i+1}", key=f"name_{i}")
        discipline = st.text_input(f"Discipline {i+1}", key=f"disc_{i}")
        material = st.text_input(f"Material {i+1}", key=f"mat_{i}")
        specification = st.text_input(f"Specification {i+1}", key=f"spec_{i}")
        quantity = st.number_input(f"Quantity {i+1}", min_value=1, value=1, key=f"qty_{i}")
        delivery_date = st.date_input(f"Delivery Date {i+1}", key=f"date_{i}")

        equipment_list.append({
            "Equipment Name": name,
            "Discipline": discipline,
            "Material": material,
            "Specification": specification,
            "Quantity": quantity,
            "Delivery Date": delivery_date
        })

    if st.button("Save Datasheet"):
        df = pd.DataFrame(equipment_list)
        save_datasheet(df)
        st.success("Datasheet saved successfully!")
        st.dataframe(df)
