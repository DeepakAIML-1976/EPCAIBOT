import os
import pandas as pd

# Create directories if not exist
os.makedirs("data/datasheets", exist_ok=True)
os.makedirs("data/rfq", exist_ok=True)

def save_datasheet(df):
    """
    Save a datasheet as a CSV with versioning.
    """
    filename = f"data/datasheets/datasheet_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

def save_rfq_file(uploaded_file, filename):
    """
    Save uploaded RFQ file to the RFQ directory.
    """
    file_path = os.path.join("data/rfq", filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
