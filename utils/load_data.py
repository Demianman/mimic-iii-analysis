"""
utils/load_data.py
------------------
Load and cache MIMIC-III CSV files.
Set DATA_DIR to the folder containing your CSV files.
"""

import os
import pandas as pd
import streamlit as st

# ── Set this to your CSV folder path ─────────────────────────────────────────
DATA_DIR = os.environ.get("MIMIC_DATA_DIR", "./data")


@st.cache_data
def load(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"File not found: {path}\nSet MIMIC_DATA_DIR or place CSVs in ./data/")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_all() -> dict:
    files = {
        "patients":      "PATIENTS.csv",
        "admissions":    "ADMISSIONS.csv",
        "icustays":      "ICUSTAYS.csv",
        "labevents":     "LABEVENTS.csv",
        "chartevents":   "CHARTEVENTS.csv",
        "prescriptions": "PRESCRIPTIONS.csv",
        "diagnoses":     "DIAGNOSES_ICD.csv",
    }
    return {key: load(fname) for key, fname in files.items()}
