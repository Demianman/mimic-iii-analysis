"""
app.py — Entry point
"""
import streamlit as st

st.set_page_config(
    page_title="MIMIC-III Analysis",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 MIMIC-III Clinical Data Analysis")
st.caption("Data Insights AI Intern Interview — Demo Prototype")

st.markdown("""
## Navigation

Use the sidebar to navigate between levels:

| Page | Content |
|------|---------|
| **Level 1 — Exploration** | Dataset overview, mortality stats, diagnoses, ICU LOS, lab distributions, ERD |
| **Level 2 — Mortality Model** | Logistic Regression vs Random Forest vs Gradient Boosting · Feature importance · ROC |
| **Level 3 — eDISH** | Pharmacovigilance · Hy's Law · KDIGO renal staging · Multi-organ safety story |

---

## Setup

1. Place your MIMIC-III CSV files in a `data/` folder (or set `MIMIC_DATA_DIR`)
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

---
*Built with Python · Streamlit · Plotly · scikit-learn · Claude API*
""")
