# Beone-note

# MIMIC-III Analysis Notes

## What This Is

A clinical decision support prototype built on the MIMIC-III demo dataset. MIMIC-III (Medical Information Mart for Intensive Care) is an open-access anonymized database covering critical care admissions at Beth Israel Deaconess Medical Center, 2001-2012. This demo subset covers approximately 100 patients, 129 admissions, and 136 ICU stays across 26 CSV files.

The central question driving the analysis: which patients should a physician prioritize first thing in the morning, and what signals in the data tell us that earliest?

---

## Data Architecture

The dataset is split across multiple relational tables. No single file answers any meaningful clinical question on its own — everything requires joining.

The key hierarchy is:

```
subject_id  →  identifies the person  (PATIENTS.csv)
hadm_id     →  identifies one hospital visit  (ADMISSIONS.csv)
icustay_id  →  identifies one ICU stay within that visit  (ICUSTAYS.csv)
```

Each analysis draws from different tables:

- Mortality rate comes from `ADMISSIONS.csv`, specifically the `hospital_expire_flag` column
- Top diagnoses come from `DIAGNOSES_ICD.csv`, joined back to admissions via `hadm_id`
- ICU length of stay and care unit come from `ICUSTAYS.csv`
- Lab values come from `LABEVENTS.csv`, filtered by `itemid` and time window
- Age requires joining `PATIENTS.csv` to `ADMISSIONS.csv` on `subject_id`

The `subject_id` and `hadm_id` columns are the glue connecting everything. Tables like `D_LABITEMS.csv` and `D_ITEMS.csv` are dictionaries — they translate numeric item IDs into human-readable lab names.

---

## Technical Issue Encountered: Date Overflow Bug

**What happened:** calculating patient age using `.dt.days` caused an integer overflow error.

**Why:** MIMIC intentionally shifts the date of birth of patients aged 89 and older back to the 1800s as part of its de-identification process. When pandas tries to compute the time difference between a 2100s admission date and an 1800s birth date using nanosecond-level int64 arithmetic, it overflows.

**Fix:** subtract years directly instead of computing a timedelta.

```python
# Wrong — overflows on MIMIC age-shifted patients
age = (admittime - dob).dt.days / 365.25

# Correct
age = admittime.dt.year - dob.dt.year
```

This is worth mentioning in the demo because it demonstrates awareness of how de-identification affects downstream computation — not just a bug fix, but understanding why it happened.

---

## Missing Data: Clinical Context Matters

Missing values in clinical data are fundamentally different from missing values in most other datasets. The reason something is missing often carries information.

Three distinct causes of missingness in MIMIC:

1. The test was not ordered — a patient with no lactate measurement is probably less critically ill than one with ten draws. The absence is a signal.
2. Short ICU stay — some patients were discharged or died before accumulating many measurements.
3. Data entry — some values were recorded as free text rather than numeric fields.

Current approach in this prototype is median imputation, which is the most conservative safe default. The alternatives considered were:

- Explicit missingness flag: add a binary column `creatinine_missing = 1` before imputing. This preserves the signal that no test was ordered.
- Zero imputation for lab counts: treat an unordered test as baseline normal rather than average-sick.
- Dropping patients with excessive missing labs: not viable here given only 129 admissions.

The same logic applies to `INPUTEVENTS_CV.csv`, where the `rate` column is null for bolus administrations. That null does not mean the data is missing — it means the drug was given all at once rather than as a continuous drip. Treating it as a missing value would be incorrect.

The principle: before imputing anything, ask why it is missing.

---

## Level 1 — Exploration

**The three findings that go beyond basic EDA:**

Finding 1 — Emergency admissions and ICU acuity. Crossing `admission_type` from ADMISSIONS with `los` from ICUSTAYS shows whether emergency patients stay longer in the ICU. If they do, admission type becomes a meaningful risk stratifier rather than just an administrative field, and it should be included as a model feature.

Finding 2 — Care unit specialization. Joining DIAGNOSES_ICD with ICUSTAYS on `hadm_id` and comparing diagnosis distributions across MICU, CCU, and SICU reveals whether units treat meaningfully different patient populations. This has practical implications for staffing and protocol design.

Finding 3 — Lab draw frequency as an acuity proxy. Counting the number of lab draws per patient in their first 24 hours and plotting against mortality outcome surfaces a non-obvious signal: patients who died tended to have more lab draws, because clinicians ordered more tests when they were worried. This motivates adding `n_lab_draws_24h` as a feature in Level 2.

---

## Level 2 — Modeling

**The outcome variable** is `hospital_expire_flag` in ADMISSIONS — binary, 0 = survived, 1 = died in hospital. That is y.

**Feature groups:**

```
ADMISSIONS       ->  age, gender, admission_type   (who is this patient)
DIAGNOSES_ICD    ->  Charlson Comorbidity Index    (how sick before this visit)
LABEVENTS        ->  first 24h lab values          (how sick right now)
```

The extraction challenge for lab features is the time join — filtering LABEVENTS to only rows where `charttime` falls within 24 hours of `admittime`. This window was chosen deliberately: it simulates the information available to a physician on the morning after admission, which is the point in time where proactive intervention is most valuable.

**Why three models instead of one:**

Logistic Regression is the interpretable baseline. Its coefficients directly quantify the relationship between each feature and mortality risk. In clinical settings, explainability often matters more than marginal accuracy gains.

Random Forest captures non-linear interactions that logistic regression misses — for example, high creatinine may only signal mortality when combined with low hemoglobin.

Gradient Boosting typically achieves the strongest raw performance but is the least interpretable of the three. It is included for comparison but would require SHAP analysis before clinical deployment.

5-fold cross-validation was used instead of a single train/test split because with 129 admissions, a single split produces unreliable AUC estimates. CV gives stable estimates with visible variance.

**New features added beyond the brief:**

- `n_lab_draws_24h` — lab draw frequency as acuity proxy, motivated by Level 1 Finding 3
- `creatinine_trend` — last minus first creatinine value in 24h, because trajectory is more informative than a static value
- `n_diagnoses` — total ICD-9 codes per admission, as a secondary comorbidity signal

**How to read the AUC:**

Rather than stating the number alone, frame it: an AUC of 0.78 means that if you randomly pick one patient who died and one who survived, the model correctly ranks the dying patient as higher risk 78% of the time. The interesting question is then which features drive that ranking — and whether the answer makes clinical sense.

**The three findings beyond the metrics:**

Finding 1 — Feature importance by group. Coloring feature importance bars by category (demographics, comorbidity, lab values, new features) reveals whether the model is learning from baseline illness burden or from acute signals. If creatinine ranks highly, early kidney dysfunction is the dominant warning sign.

Finding 2 — False negative analysis. The patients the model missed — predicted to survive but died — are the most clinically dangerous errors. Examining their average age, Charlson score, and lab draw count reveals what the model systematically underweights.

Finding 3 — Charlson overlap by outcome. If the Charlson score distributions for survivors and non-survivors overlap heavily, it means baseline comorbidity alone does not explain mortality. The 24-hour lab values are doing the actual predictive work. This validates the multi-feature approach.

---

## Level 3 — Pharmacovigilance

The eDISH plot is an FDA-standard visualization for detecting drug-induced liver injury (DILI) signals in clinical trial data. Axes are peak ALT and peak total bilirubin, both normalized to their upper limits of normal (ULN). Hy's Law defines the danger zone: ALT above 3x ULN and bilirubin above 2x ULN simultaneously.

**Three findings beyond the eDISH plot:**

Finding 1 — Hy's Law and mortality. Connecting the eDISH flag back to `hospital_expire_flag` tests whether the pharmacovigilance signal is clinically meaningful. If Hy's Law patients have materially higher mortality, the signal warrants real intervention.

Finding 2 — Lab storm detection. Flagging patients with three or more organ systems simultaneously in distress (liver, renal, hematological, lactate, WBC) and plotting mortality rate against the number of organs flagged produces the clearest finding in the dataset: each additional organ system in distress substantially increases mortality risk. This is the foundation of the proactive alerting concept.

Finding 3 — Patient lab profile heatmap. One row per patient, columns are key labs normalized to ULN, color encodes abnormality, rows sorted by outcome. This is the closest thing in the prototype to a physician-facing tool — a visual that makes the difference between high-risk and low-risk patients immediately apparent without requiring any interpretation of numbers.

---

## The Connecting Narrative

The three levels are not three separate analyses. They answer one question in sequence.

Level 1 establishes who these patients are and surfaces early signals — admission type, care unit patterns, lab draw intensity.

Level 2 builds a model that formalizes those signals into a risk score, then interrogates the model to understand what it is actually learning.

Level 3 shows what organ-level damage looks like in the data, and demonstrates that multi-organ involvement is the strongest observable predictor of death.

Together they answer: which patients should a physician look at first, and what in the data tells us that before the patient deteriorates visibly.

---

## What Would Come Next

- SOFA score computed from CHARTEVENTS — a more validated ICU severity measure than the Charlson Index
- Time-series lab trend features replacing static 24-hour summaries
- Drug-lab linkage in the eDISH analysis — connecting PRESCRIPTIONS timestamps to liver enzyme spikes to identify which agents precede injury
- SHAP values for individual patient explainability, not just global feature importance
- A real-time alert threshold engine that fires when a patient's 6-hour lab trend crosses a defined danger zone