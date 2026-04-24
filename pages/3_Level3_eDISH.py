"""
pages/3_Level3_eDISH.py
------------------------
Level 3: Pharmacovigilance — eDISH scatter plot + Hy's Law + lab storytelling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_all

st.set_page_config(page_title="Level 3 — eDISH", layout="wide")
st.title("⚗️ Level 3 — Pharmacovigilance Signal Detection")
st.caption("eDISH analysis · Hy's Law · Lab safety storytelling")

with st.spinner("Loading data..."):
    data = load_all()

labevents   = data["labevents"].copy()
admissions  = data["admissions"].copy()
patients    = data["patients"].copy()

labevents.columns  = labevents.columns.str.lower()
admissions.columns = admissions.columns.str.lower()
patients.columns   = patients.columns.str.lower()

labevents["valuenum"] = pd.to_numeric(labevents["valuenum"], errors="coerce")
labevents["charttime"] = pd.to_datetime(labevents["charttime"])

# ── ULN constants ─────────────────────────────────────────────────────────────
ALT_ITEM    = 50861   # ALT itemid
BILI_ITEM   = 50885   # Total bilirubin
CREAT_ITEM  = 50912   # Creatinine
WBC_ITEM    = 51301
HGB_ITEM    = 51222
PLT_ITEM    = 51265

ALT_ULN   = 40.0    # U/L
BILI_ULN  = 1.2     # mg/dL
CREAT_ULN = 1.2     # mg/dL

# ── Peak values per patient ───────────────────────────────────────────────────
def peak(item_id, col_name):
    sub = labevents[labevents["itemid"] == item_id].dropna(subset=["valuenum"])
    return sub.groupby("subject_id")["valuenum"].max().rename(col_name)

alt_peak   = peak(ALT_ITEM,   "peak_alt")
bili_peak  = peak(BILI_ITEM,  "peak_bili")
creat_peak = peak(CREAT_ITEM, "peak_creat")
wbc_peak   = peak(WBC_ITEM,   "peak_wbc")
hgb_min    = labevents[labevents["itemid"]==HGB_ITEM].dropna(subset=["valuenum"]).groupby("subject_id")["valuenum"].min().rename("min_hgb")
plt_min    = labevents[labevents["itemid"]==PLT_ITEM].dropna(subset=["valuenum"]).groupby("subject_id")["valuenum"].min().rename("min_plt")

df_safety = pd.concat([alt_peak, bili_peak, creat_peak, wbc_peak, hgb_min, plt_min], axis=1).reset_index()
df_safety = df_safety.merge(admissions[["subject_id","hospital_expire_flag","admission_type"]].drop_duplicates("subject_id"), on="subject_id", how="left")

# Normalise to ULN
df_safety["alt_uln"]   = df_safety["peak_alt"]   / ALT_ULN
df_safety["bili_uln"]  = df_safety["peak_bili"]  / BILI_ULN
df_safety["creat_uln"] = df_safety["peak_creat"] / CREAT_ULN

# Hy's Law: ALT > 3× ULN AND Bili > 2× ULN
df_safety["hys_law"] = (df_safety["alt_uln"] > 3) & (df_safety["bili_uln"] > 2)

# KDIGO staging (creatinine-based, simplified)
def kdigo(creat_uln_val):
    if pd.isna(creat_uln_val): return "Unknown"
    if creat_uln_val >= 3.0:   return "Stage 3"
    if creat_uln_val >= 2.0:   return "Stage 2"
    if creat_uln_val >= 1.5:   return "Stage 1"
    return "Normal"

df_safety["kdigo_stage"] = df_safety["creat_uln"].apply(kdigo)

df_edish = df_safety.dropna(subset=["alt_uln","bili_uln"])

# ═══════════════════════════════════════════════════════════════════════════════
# eDISH PLOT
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 eDISH Scatter Plot — Hepatotoxicity Signal Detection")

col1, col2, col3 = st.columns(3)
col1.metric("Patients with liver data", len(df_edish))
col2.metric("Hy's Law cases", int(df_safety["hys_law"].sum()))
col3.metric("ALT > 3× ULN", int((df_safety["alt_uln"] > 3).sum()))

fig = go.Figure()

# Quadrant background shading
fig.add_shape(type="rect", x0=3, y0=2, x1=df_edish["alt_uln"].max()*1.1, y1=df_edish["bili_uln"].max()*1.1,
              fillcolor="rgba(231,76,60,0.12)", line_width=0)
fig.add_shape(type="rect", x0=0, y0=0, x1=3, y1=2,
              fillcolor="rgba(46,204,113,0.08)", line_width=0)

# Normal patients
normal = df_edish[~df_edish["hys_law"]]
fig.add_trace(go.Scatter(
    x=normal["alt_uln"], y=normal["bili_uln"],
    mode="markers", name="Normal",
    marker=dict(color="#636EFA", size=8, opacity=0.6),
    text=normal["subject_id"].astype(str),
    hovertemplate="Patient %{text}<br>ALT: %{x:.1f}×ULN<br>Bili: %{y:.1f}×ULN",
))

# Hy's Law patients
hys = df_edish[df_edish["hys_law"]]
if len(hys) > 0:
    fig.add_trace(go.Scatter(
        x=hys["alt_uln"], y=hys["bili_uln"],
        mode="markers", name="Hy's Law ⚠️",
        marker=dict(color="#e74c3c", size=14, symbol="star", line=dict(width=1, color="darkred")),
        text=hys["subject_id"].astype(str),
        hovertemplate="Patient %{text}<br>ALT: %{x:.1f}×ULN<br>Bili: %{y:.1f}×ULN<br>⚠️ Hy's Law",
    ))

# Reference lines
fig.add_vline(x=3, line_dash="dash", line_color="orange", annotation_text="3× ULN (ALT)")
fig.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="2× ULN (Bili)")
fig.add_vline(x=1, line_dash="dot", line_color="gray", annotation_text="1× ULN")
fig.add_hline(y=1, line_dash="dot", line_color="gray")

fig.update_layout(
    title="eDISH: Peak ALT vs Peak Bilirubin (normalised to ULN)",
    xaxis_title="Peak ALT / ULN (40 U/L)",
    yaxis_title="Peak Total Bilirubin / ULN (1.2 mg/dL)",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> **Hy's Law zone** (top-right, red shading): ALT > 3× ULN **AND** Bilirubin > 2× ULN.  
> Patients in this quadrant are at elevated risk of drug-induced liver injury (DILI) and warrant clinical review.
""")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# RENAL SAFETY — KDIGO
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🫀 Renal Safety — KDIGO Staging")

kdigo_counts = df_safety["kdigo_stage"].value_counts().reset_index()
kdigo_counts.columns = ["Stage", "Count"]
order = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Unknown"]
kdigo_counts["Stage"] = pd.Categorical(kdigo_counts["Stage"], categories=order, ordered=True)
kdigo_counts = kdigo_counts.sort_values("Stage")

fig = px.bar(
    kdigo_counts, x="Stage", y="Count",
    color="Stage",
    color_discrete_map={"Normal": "#2ecc71", "Stage 1": "#f1c40f", "Stage 2": "#e67e22", "Stage 3": "#e74c3c", "Unknown": "#95a5a6"},
    title="KDIGO AKI Staging from Peak Creatinine",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# LAB SAFETY STORYTELLING
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📖 Lab Safety Story — Multi-Organ Toxicity Profile")

st.markdown("""
*Connecting hepatic, renal, and hematological signals to tell a patient safety story.*
""")

# Multi-organ risk matrix
df_story = df_safety.copy()
df_story["liver_risk"]  = (df_story["alt_uln"] > 3).map({True: "High", False: "Normal"})
df_story["renal_risk"]  = df_story["kdigo_stage"].apply(lambda x: "High" if x in ["Stage 2","Stage 3"] else "Normal")
df_story["heme_risk"]   = ((df_story["min_hgb"] < 8) | (df_story["min_plt"] < 100)).map({True: "High", False: "Normal"})
df_story["organs_affected"] = (
    (df_story["liver_risk"] == "High").astype(int) +
    (df_story["renal_risk"] == "High").astype(int) +
    (df_story["heme_risk"]  == "High").astype(int)
)

organ_counts = df_story["organs_affected"].value_counts().sort_index().reset_index()
organ_counts.columns = ["Organs Affected", "Patients"]

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(
        organ_counts, x="Organs Affected", y="Patients",
        color="Organs Affected",
        color_continuous_scale="Reds",
        title="Patients by Number of Organ Systems at Risk",
        labels={"Organs Affected": "# Organ Systems (liver / renal / heme)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Mortality by organs affected
    mort = df_story.groupby("organs_affected")["hospital_expire_flag"].mean().reset_index()
    mort.columns = ["Organs Affected", "Mortality Rate"]
    mort["Mortality Rate"] = mort["Mortality Rate"] * 100
    fig = px.line(
        mort, x="Organs Affected", y="Mortality Rate",
        markers=True,
        title="Mortality Rate by Organs Affected",
        labels={"Mortality Rate": "Mortality %"},
        color_discrete_sequence=["#e74c3c"],
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Key observation:** Patients with multi-organ involvement show markedly higher mortality rates,
suggesting that combined hepatic + renal + hematological toxicity is a critical safety signal.
This pattern supports multi-biomarker monitoring protocols rather than single-organ surveillance.
""")

st.divider()
st.subheader("📝 Analyst's Log")
st.markdown(f"""
**What I built:** eDISH hepatotoxicity analysis + KDIGO renal staging + multi-organ safety story.

**Key findings:**
- **{int(df_safety['hys_law'].sum())} patients** meet Hy's Law criteria (ALT >3×ULN + Bili >2×ULN) — potential DILI signal
- KDIGO staging reveals a subset with significant AKI
- Multi-organ involvement correlates strongly with mortality

**Assumptions:**
- Peak values used (worst-case safety posture) — appropriate for pharmacovigilance
- No drug attribution possible without linking PRESCRIPTIONS timeline — would be the next step
- KDIGO staging is simplified (uses peak creatinine, not change from baseline)

**If I had more time:**
- Link PRESCRIPTIONS to identify which drugs precede lab spikes
- Add temporal analysis: how quickly do values rise/normalise?
- CTCAE grading for WBC/hemoglobin/platelets
""")
