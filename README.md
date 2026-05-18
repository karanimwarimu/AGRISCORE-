# AGRISCORE-
Alternative Credit Scoring for Smallholder Farmers

# 🌾 AgriScore — Alternative Credit Scoring for Smallholder Farmers

**Kenya & Tanzania | Machine Learning + Explainable AI (SHAP)**

---

## 1. Overview
AgriScore is a data-driven credit scoring prototype that estimates a farmer’s:
- **Probability of Default (PD)**
- **Credit Score** (scaled numeric output)
- **Risk Tier** (Low / Medium / High)
- **Recommendation** (Approve / Approve with Caution / Review-Decline)

Unlike traditional lending approaches that depend heavily on credit history, AgriScore uses **agricultural productivity signals**, **household attributes**, and **climate risk indicators** to support alternative underwriting.

The project is deployed via a **Streamlit dashboard** that supports:
- **Single farmer scoring** with SHAP-based explanation
- **Batch scoring** from uploaded CSV files
- **Global interpretability** (SHAP summary plot)

---

## 2. Final Aim: What we wanted to build — and what we got
### Final business aim
Create an **alternative credit scoring system** for **smallholder farmers** where conventional credit history is often missing or unreliable.

### What we built (deliverables)
1. **Preprocessing pipeline** that transforms raw farmer inputs into the exact feature space expected by the model.
2. **LightGBM classification model** that predicts PD.
3. **Scoring layer** that converts PD into lender-style decisions (credit score + tier + recommendation).
4. **Explainability layer (SHAP)** to show top drivers behind each prediction.
5. **Streamlit deployment** for usability (single + batch scoring, global insights).

### What we got
- A working end-to-end scoring flow: **inputs → preprocessing → PD prediction → decision outputs → explanations**.
- A deployed UI that demonstrates how stakeholders can interpret and use model outputs.

---

## 3. Business Understanding
### Why alternative credit scoring?
Smallholder farmers commonly face:
- limited formal credit history
- highly variable yields and cash flow
- climate-driven production risk
- sparse documentation for lenders

A practical alternative is to use observable signals (e.g., yield potential and climate risk proxies) as a basis for underwriting.

### How lenders can use AgriScore
- **Loan underwriting**: risk-tier applications and set appropriate approval thresholds
- **Portfolio management**: monitor risk patterns over applicant profiles
- **Risk mitigation planning**: prioritize resilience interventions (e.g., drought response)

---

## 4. Data preprocessing
### Data organization in the repo
- `Data sets/Raw data/` — source data
- `Data sets/Clean data/` — cleaned intermediate datasets
- `Data sets/Processed data/` — processed feature-ready datasets
- Training artifacts reference: `Data sets/Final_dataset.csv`

### Target variable creation (capstone approach)
The notebook uses a synthetic supervised target:
- **good_borrower = 1** when a farmer meets conditions aligned with better expected repayment (based on medians of selected agronomic/climate features)
- otherwise **good_borrower = 0**
- controlled noise is added to mimic uncertainty

### Encoding & feature alignment
`agriscore_application/preprocessing.py` implements `preprocess_for_scoring()`:
1. Loads the saved **encoder** metadata (categorical columns + feature names).
2. One-hot encodes categorical features.
3. Ensures all features are **numeric** (coercion + fallbacks).
4. Forces the final DataFrame to match the training schema defined in `agriscore_application/config.py`:
   - `FEATURE_COLUMNS`
5. Fills missing feature columns with zeros.

**Deployment artifacts used by preprocessing**:
- `Models/encoder.pkl`
- `agriscore_application/config.py` (authoritative feature schema)

---

## 5. Model development & evaluation
### Modeling workflow
In `Notebooks/Models.ipynb`, multiple baselines were trained and compared:
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM (chosen final model)

Class imbalance was addressed using **SMOTE** during training.

### Final model selection: LightGBM
The LightGBM model was selected for its strong generalization and balanced performance compared to other models.

### Evaluation metrics (used across models)
The notebook evaluates with:
- **Accuracy**
- **Precision / Recall / F1-score**
- **ROC-AUC**
- Confusion matrices
- ROC curves and Precision–Recall curves

### Hyperparameter tuning
LightGBM was tuned using:
- **Stratified K-Fold** cross-validation
- **RandomizedSearchCV**
- optimization focused on **F1-macro**

### Reported results (from notebook narrative)
The notebook states LightGBM performed best overall, with emphasis on:
- strong AUC-ROC
- improved classification metrics

(These notebook-reported values are the source of the project’s model-selection rationale.)

---

## 6. Explainability (SHAP)
AgriScore uses SHAP to make the model’s decisions interpretable.

### What explanations show
- **Global**: which features matter most overall
- **Local**: which features drove an individual farmer’s decision

### Key implementation
In `agriscore_application/explainer.py`:
- `load_explainer()` loads `Models/agriscore_SHAP_explainer.pkl`
- `get_explanation()` returns:
  - top positive SHAP drivers
  - top negative SHAP drivers
- `plot_shap_summary()` generates and saves a global SHAP summary plot

The Streamlit UI additionally renders an individual **SHAP force plot** for single predictions.

---

## 7. Deployment (Streamlit dashboard)
### Streamlit features
`agriscore_application/streamlit_app.py` provides three modes:
1. **Single Farmer Scoring**
   - user inputs agricultural + household parameters
   - outputs:
     - credit score
     - risk tier
     - PD
     - recommendation
     - explanation (top SHAP drivers + force plot)
2. **Batch Scoring (CSV)**
   - upload CSV → score all farmers → show table + download results
3. **Global Insights**
   - button to generate SHAP summary plot and display it

### Assets expected in the repo
- `Models/lightgbm_model.pkl`
- `Models/agriscore_SHAP_explainer.pkl`
- `Models/encoder.pkl`

---

## 8. Folder structure

```text
agriscoreproject/
├─ Notebooks/
│  └─ Models.ipynb
├─ Proposal/
│  ├─ AgriScore Project Work Plan.pdf
│  └─ ...
├─ Data sets/
│  ├─ Raw data/
│  ├─ Clean data/
│  └─ Processed data/
├─ Models/
│  ├─ lightgbm_model.pkl
│  ├─ encoder.pkl
│  ├─ feature_columns.pkl
│  └─ agriscore_SHAP_explainer.pkl
├─ Visuals/
│  ├─ shap_summary.png
│  └─ ...
└─ agriscore_application/
   ├─ config.py
   ├─ preprocessing.py
   ├─ models.py
   ├─ scoring.py
   ├─ explainer.py
   ├─ shap_exe.py
   ├─ setup_encoder.py
   ├─ streamlit_app.py
   └─ test.py / test2.py
```

---

## 9. Problems faced (and how we handled them)
1. **Feature/schema mismatches at deployment**
   - fixed by forcing the preprocessing output to match `FEATURE_COLUMNS`.

2. **Categorical handling during inference**
   - fixed by saving and reusing the encoder artifact (`Models/encoder.pkl`).

3. **SHAP output shape differences**
   - handled by treating SHAP values as either a list (multi-class style) or an array.

4. **Imbalanced dataset**
   - addressed with **SMOTE** in the modeling notebook.

---

## 10. Technologies used
- **Python**
- **Streamlit** (dashboard)
- **LightGBM** (final model)
- **SHAP** (explainability)
- **Pandas / NumPy** (data handling)
- **scikit-learn** (pipelines, metrics, model selection)
- **OneHotEncoder** (categorical encoding)
- **joblib** (model + encoder + explainer persistence)
- **Matplotlib** (plots)

---

## 11. Streamlit dashboard link
Dashboard deployed at:
**URL: [ADD_STREAMLIT_URL_HERE]**

---

## 12. Future improvements (what we want to target next)
1. **More realistic target definition**
   - replace synthetic label generation with a repayment/default proxy that better reflects ground truth.

2. **Better evaluation strategy**
   - calibration (e.g., reliability curves / Platt or isotonic calibration for PD)
   - threshold optimization for the business objective.

3. **Richer feature engineering**
   - incorporate time-series features (past yields, rainfall trends, historical farm performance)
   - incorporate weather/drought indicators more directly.

4. **Fairness & robustness checks**
   - validate performance stability across regions / crop types.

5. **Operational deployment hardening**
   - stricter input schema validation in Streamlit
   - clearer handling of missing/unseen categories.

6. **Impact analysis (cost–benefit)**
   - evaluate how approval strategies affect default rates and lender outcomes.

---

## 13. How to run locally
1. Ensure dependencies are installed (see project requirements in your environment).
2. Run:

```bash
streamlit run agriscore_application/streamlit_app.py
```

3. Use the dashboard modes to score single or batch inputs.


