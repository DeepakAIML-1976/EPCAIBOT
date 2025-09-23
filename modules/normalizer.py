# modules/normalizer.py

# A minimal mapping table for common datasheet fields -> normalized keys
ATTRIBUTE_MAPPING = {
    "Equipment Name": "EQUIPMENT_NAME",
    "Equipment": "EQUIPMENT_NAME",
    "Model": "MODEL",
    "Type": "MODEL",
    "Material": "MATERIAL",
    "Material Grade": "MATERIAL",
    "Pressure Rating": "PRESSURE_RATING",
    "Rating": "PRESSURE_RATING",
    "Quantity": "QUANTITY",
    "Qty": "QUANTITY",
    "Delivery Date": "DELIVERY_DATE",
    "Temperature": "TEMPERATURE",
    "Power": "POWER",
}


def normalize_key(k: str) -> str:
    if not k:
        return k
    # basic normalization
    k_clean = k.strip().title()
    if k_clean in ATTRIBUTE_MAPPING:
        return ATTRIBUTE_MAPPING[k_clean]
    # fallback: uppercase alphanumeric
    return k.upper().replace(" ", "_")
    
    
def normalize_kv_dict(kv: dict) -> dict:
    """
    Return normalized dict mapping to normalized keys.
    """
    out = {}
    for k, v in kv.items():
        nk = normalize_key(k)
        out[nk] = v
    return out
