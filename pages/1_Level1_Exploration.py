"""
pages/1_Level1_Exploration.py
------------------------------
Level 1: Schema exploration dashboard + ERD
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_all

st.set_page_config(page_title="Level 1 — Exploration", layout="wide")
st.title("🏥 Level 1 — Exploration & Understanding")
st.caption("MIMIC-III Demo Dataset · ICU patients from Beth Israel Deaconess Medical Center")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading MIMIC-III tables..."):
    data = load_all()

patients    = data["patients"]
admissions  = data["admissions"]
icustays    = data["icustays"]
labevents   = data["labevents"]
diagnoses   = data["diagnoses"]

if admissions.empty:
    st.error("Data not found. Set MIMIC_DATA_DIR environment variable to your CSV folder.")
    st.stop()

# ── Clean column names ────────────────────────────────────────────────────────
admissions.columns  = admissions.columns.str.lower()
patients.columns    = patients.columns.str.lower()
icustays.columns    = icustays.columns.str.lower()
labevents.columns   = labevents.columns.str.lower()
diagnoses.columns   = diagnoses.columns.str.lower()

# ── Merge patients + admissions ───────────────────────────────────────────────
adm = admissions.merge(patients[["subject_id","dob","gender"]], on="subject_id", how="left")
adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
adm["dob"]       = pd.to_datetime(adm["dob"], errors="coerce")
# MIMIC shifts DOB for patients >89 to ~1800s — use year subtraction to avoid int64 overflow
adm["age"] = (adm["admittime"].dt.year - adm["dob"].dt.year).clip(0, 100).fillna(65)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Key metrics
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Dataset Overview")

total_patients   = adm["subject_id"].nunique()
total_admissions = len(adm)
total_deaths     = adm["hospital_expire_flag"].sum()
mortality_rate   = total_deaths / total_admissions * 100
avg_icu_los      = icustays["los"].mean() if "los" in icustays.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unique Patients",  total_patients)
c2.metric("Total Admissions", total_admissions)
c3.metric("In-Hospital Deaths", int(total_deaths))
c4.metric("Mortality Rate",   f"{mortality_rate:.1f}%")
c5.metric("Avg ICU LOS (days)", f"{avg_icu_los:.1f}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Mortality & Admission type
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown("**In-Hospital Mortality**")
    mort_counts = adm["hospital_expire_flag"].value_counts().reset_index()
    mort_counts.columns = ["outcome", "count"]
    mort_counts["outcome"] = mort_counts["outcome"].map({0: "Survived", 1: "Died"})
    fig = px.pie(
        mort_counts, values="count", names="outcome",
        color="outcome",
        color_discrete_map={"Survived": "#2ecc71", "Died": "#e74c3c"},
        hole=0.45,
    )
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Mortality Rate by Admission Type**")
    mort_by_type = adm.groupby("admission_type").agg(
        total=("hospital_expire_flag","count"),
        deaths=("hospital_expire_flag","sum")
    ).reset_index()
    mort_by_type["mortality_pct"] = (mort_by_type["deaths"] / mort_by_type["total"] * 100).round(1)
    fig = px.bar(
        mort_by_type, x="admission_type", y="mortality_pct",
        text="mortality_pct",
        color="mortality_pct",
        color_continuous_scale="Reds",
        labels={"admission_type": "Admission Type", "mortality_pct": "Mortality %"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Top diagnoses + ICU LOS by care unit
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 Most Common Diagnoses (ICD-9)**")
    top_dx = diagnoses["icd9_code"].value_counts().head(10).reset_index()
    top_dx.columns = ["icd9_code", "count"]
    fig = px.bar(
        top_dx, x="count", y="icd9_code", orientation="h",
        color="count", color_continuous_scale="Blues",
        labels={"icd9_code": "ICD-9 Code", "count": "Count"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**ICU Length of Stay by Care Unit**")
    if "first_careunit" in icustays.columns and "los" in icustays.columns:
        icu_clean = icustays[icustays["los"] < 30]  # exclude extreme outliers for viz
        fig = px.box(
            icu_clean, x="first_careunit", y="los",
            color="first_careunit",
            labels={"first_careunit": "Care Unit", "los": "LOS (days)"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        median_los = icu_clean["los"].median()
        fig.add_hline(y=median_los, line_dash="dash", line_color="red",
                      annotation_text=f"Overall median: {median_los:.1f}d")
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Lab distributions + ICU LOS histogram
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Lab Value Distributions (Key Labs)**")
    # itemids: creatinine=50912, WBC=51301, hemoglobin=51222, platelets=51265
    lab_map = {50912: "Creatinine", 51301: "WBC", 51222: "Hemoglobin", 51265: "Platelets"}
    lab_subset = labevents[labevents["itemid"].isin(lab_map.keys())].copy()
    lab_subset["lab_name"] = lab_subset["itemid"].map(lab_map)
    lab_subset["valuenum"] = pd.to_numeric(lab_subset["valuenum"], errors="coerce")
    lab_subset = lab_subset.dropna(subset=["valuenum"])

    # Clip extreme outliers per lab for readability
    def clip_99(g):
        lo, hi = g["valuenum"].quantile(0.01), g["valuenum"].quantile(0.99)
        return g[(g["valuenum"] >= lo) & (g["valuenum"] <= hi)]
    lab_clean = lab_subset.groupby("lab_name", group_keys=False).apply(clip_99)

    fig = px.histogram(
        lab_clean, x="valuenum", facet_col="lab_name",
        facet_col_wrap=2, nbins=40,
        color="lab_name",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"valuenum": "Value"},
    )
    fig.update_layout(showlegend=False, height=400, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**ICU Length of Stay Distribution**")
    icu_los = icustays[icustays["los"].between(0, 30)]["los"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=icu_los, nbinsx=40, marker_color="#636EFA", name="LOS"))
    fig.add_vline(
        x=icu_los.median(), line_dash="dash", line_color="red",
        annotation_text=f"Median: {icu_los.median():.1f}d",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="LOS (days)", yaxis_title="Count",
        margin=dict(t=20), height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ERD
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ Entity Relationship Diagram")

erd_html = """
<div style="font-family: monospace; background: #1e1e2e; color: #cdd6f4;
            padding: 24px; border-radius: 12px; overflow-x: auto; font-size: 13px;">
<pre>
┌─────────────────────┐          ┌──────────────────────────┐
│      PATIENTS       │          │       ADMISSIONS          │
│─────────────────────│          │──────────────────────────│
│ PK subject_id       │──────────│ PK hadm_id               │
│    gender           │  1 : N   │ FK subject_id            │
│    dob              │          │    admittime / dischtime  │
│    dod              │          │    admission_type         │
└─────────────────────┘          │    hospital_expire_flag   │
                                 └────────────┬─────────────┘
                                              │ 1 : N
              ┌───────────────────────────────┼───────────────────────────┐
              │                               │                           │
              ▼                               ▼                           ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│      ICUSTAYS       │     │      LABEVENTS        │     │    DIAGNOSES_ICD      │
│─────────────────────│     │──────────────────────│     │───────────────────────│
│ PK icustay_id       │     │ FK subject_id        │     │ FK subject_id         │
│ FK hadm_id          │     │ FK hadm_id           │     │ FK hadm_id            │
│ FK subject_id       │     │    itemid            │     │    icd9_code          │
│    first_careunit   │     │    charttime         │     │    seq_num            │
│    los              │     │    valuenum          │     └───────────────────────┘
└─────────────────────┘     └──────────────────────┘

              ┌──────────────────────────────────────────┐
              │            CHARTEVENTS                   │
              │──────────────────────────────────────────│
              │ FK subject_id · FK hadm_id · FK icustay_id│
              │    itemid · charttime · valuenum          │
              └──────────────────────────────────────────┘

              ┌──────────────────────────────────────────┐
              │            PRESCRIPTIONS                 │
              │──────────────────────────────────────────│
              │ FK subject_id · FK hadm_id · FK icustay_id│
              │    drug · startdate · enddate · dose_val  │
              └──────────────────────────────────────────┘
</pre>
<p style="color:#a6e3a1; margin-top:8px;">
  Primary key flow: PATIENTS → ADMISSIONS → ICUSTAYS / LABEVENTS / CHARTEVENTS / PRESCRIPTIONS / DIAGNOSES_ICD
</p>
</div>
"""
st.html(erd_html)

st.markdown("""
**Key relationships:**
- One patient (`subject_id`) → many admissions (`hadm_id`)
- One admission → many ICU stays, lab events, chart events, prescriptions, diagnoses
- `icustay_id` links ICU-specific events in CHARTEVENTS
""")
