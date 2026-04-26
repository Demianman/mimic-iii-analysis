# 🏥 MIMIC-III Clinical Data Analysis

> Data Insights AI Summer Intern — Interview Demo Prototype  
> Built with Python · Streamlit · Plotly · scikit-learn

---

## What this does

An end-to-end interactive dashboard covering all 3 levels of the MIMIC-III project guide:

| Level | Focus | Output |
|-------|-------|--------|
| 1 | Exploration & ERD | 6 interactive charts + entity relationship diagram |
| 2 | Mortality Prediction | 3-model comparison (LR vs RF vs GBM) + ROC + feature importance |
| 3 | Pharmacovigilance | eDISH scatter · Hy's Law · KDIGO renal staging · multi-organ story |

---

## Quick start

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/mimic-iii-analysis
cd mimic-iii-analysis

# 2. Place CSV files
mkdir data
# copy all 26 MIMIC-III CSVs into ./data/

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

---

## Project structure

```
mimic-iii-analysis/
├── app.py                          ← Entry point
├── pages/
│   ├── 1_Level1_Exploration.py     ← Dashboard + ERD
│   ├── 2_Level2_Mortality_Model.py ← ML models
│   └── 3_Level3_eDISH.py          ← Pharmacovigilance
├── utils/
│   └── load_data.py               ← Cached data loader
├── requirements.txt
└── README.md
```

---

## Key design decisions

| Decision | Reasoning |
|----------|-----------|
| 3 models compared (not just logistic regression) | Goes beyond the brief — shows initiative |
| 5-fold cross-validation | Small dataset (~129 admissions) — CV gives more reliable AUC than train/test split |
| Median imputation for missing labs | Conservative — avoids dropping patients with partial lab data |
| Peak values for safety analysis | Worst-case posture appropriate for pharmacovigilance |
| Analyst's Log on every page | Full audit trail of assumptions and decisions |

---

## If I had more time

- [ ] SOFA score from CHARTEVENTS (better ICU severity proxy than Charlson alone)
- [ ] SHAP values for individual patient explainability
- [ ] Time-series lab trends (not just 24h summaries)
- [ ] Drug-lab linkage in eDISH (which prescriptions precede liver spikes?)
- [ ] Natural language query interface over the dataset

---

*Candidate: Yiman · Data Insights AI Intern*
