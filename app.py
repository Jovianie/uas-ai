import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ========================
# CSS (diperbaiki kontras & font)
# ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300;1,9..40,400&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --bg:        #F4EFE6;
  --surface:   #FDFAF4;
  --border:    #DDD4C0;
  --accent:    #7C6A52;
  --accent2:   #A08060;
  --ink:       #1E1712;
  --ink2:      #4A3F32;
  --ink3:      #8B7B66;
  --risk-low:  #3D6E55;
  --risk-mid:  #9C7A2A;
  --risk-hi:   #8C3A35;
  --pos:       #8C3A35;
  --neg:       #3D6E55;
  --serif:     'DM Serif Display', Georgia, serif;
  --sans:      'DM Sans', system-ui, sans-serif;
}

html, body, [class*="css"], .stApp {
  background-color: var(--bg) !important;
  font-family: var(--sans) !important;
  color: var(--ink) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 1.5rem 2rem 2rem !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}

/* HEADER */
.app-masthead {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  padding: 1.5rem 1.8rem;
  background: var(--ink);
  border-radius: 6px;
  margin-bottom: 1.8rem;
}
.masthead-left { z-index: 1; }
.masthead-eyebrow {
  font-family: var(--sans);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent2);
  margin-bottom: 0.4rem;
}
.masthead-title {
  font-family: var(--serif);
  font-size: 1.8rem;
  font-weight: 400;
  color: #FAF6EF;
  line-height: 1.1;
}
.masthead-title em { font-style: italic; color: var(--accent2); }
.masthead-right { text-align: right; }
.masthead-sub {
  font-family: var(--sans);
  font-size: 0.7rem;
  color: rgba(250,246,239,0.55);
  font-weight: 300;
  max-width: 250px;
}

/* SECTION LABEL */
.sec-label {
  font-family: var(--sans);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--ink3);
  margin-bottom: 0.8rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}

/* SLIDER & SELECTBOX */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label {
  font-family: var(--sans) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  color: var(--ink2) !important;
  margin-bottom: 0.2rem !important;
}
div[data-testid="stSlider"] .st-emotion-cache-1dp5vir {
  color: var(--accent) !important;
}
.stSelectbox > div > div {
  border-color: var(--border) !important;
  border-radius: 3px !important;
  background: var(--surface) !important;
  font-family: var(--sans) !important;
  font-size: 0.78rem !important;
}

/* BUTTON */
.stButton > button {
  font-family: var(--sans) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  background: var(--ink) !important;
  color: #FAF6EF !important;
  border: none !important;
  border-radius: 3px !important;
  padding: 0.5rem 1.2rem !important;
  width: 100% !important;
  transition: 0.2s;
}
.stButton > button:hover {
  background: var(--accent) !important;
}

/* RESULT BLOCK (kontras terang) */
.result-prob {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  margin-top: 0.5rem;
}
.prob-number {
  font-family: var(--serif);
  font-size: 2.8rem;
  font-weight: 400;
  color: var(--ink);
  line-height: 1;
}
.risk-badge {
  display: inline-block;
  font-family: var(--sans);
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.3rem 0.9rem;
  border-radius: 2px;
}
.risk-low  { background: #E8F2EC; color: var(--risk-low); border: 1px solid #B8D9C4; }
.risk-med  { background: #F5EDD5; color: var(--risk-mid); border: 1px solid #DFC98A; }
.risk-hi   { background: #F2E4E3; color: var(--risk-hi); border: 1px solid #D9A8A6; }

/* FACTOR TABLE */
.factor-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--border);
}
.factor-name { flex: 2; font-size: 0.75rem; font-weight: 500; color: var(--ink2); }
.factor-bar { flex: 3; height: 4px; background: #EDE6D8; border-radius: 2px; overflow: hidden; margin: 0 0.8rem; }
.factor-bar-fill { height: 100%; border-radius: 2px; }
.factor-pct { width: 40px; text-align: right; font-size: 0.7rem; font-weight: 600; }
.factor-dir { width: 22px; text-align: center; font-weight: 700; font-size: 0.7rem; }
.pos-fill { background: var(--pos); }
.neg-fill { background: var(--neg); }
.pos-dir { color: var(--pos); }
.neg-dir { color: var(--neg); }

.legend {
  display: flex;
  gap: 1rem;
  margin-top: 0.6rem;
  padding-top: 0.4rem;
  font-size: 0.62rem;
  color: var(--ink3);
  border-top: 1px solid var(--border);
}
.legend-dot { width: 6px; height: 6px; border-radius: 1px; display: inline-block; margin-right: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ========================
# Constants
# ========================
NUMERIC_FEATURES = [
    "Study_Hours_per_Day", "Attendance_Rate", "Assignment_Delay_Days",
    "Travel_Time_Minutes", "Stress_Index", "GPA",
]
CAT_FEATURES = ["Internet_Access", "Part_Time_Job", "Scholarship", "Semester"]
INPUT_COLS = NUMERIC_FEATURES + CAT_FEATURES

OHE_LABELS = {
    "Internet_Access_No":  "No Internet Access",
    "Internet_Access_Yes": "Internet Access Available",
    "Part_Time_Job_No":    "No Part-Time Job",
    "Part_Time_Job_Yes":   "Has Part-Time Job",
    "Scholarship_No":      "No Scholarship",
    "Scholarship_Yes":     "Has Scholarship",
    "Semester_Year 1":     "Year 1",
    "Semester_Year 2":     "Year 2",
    "Semester_Year 3":     "Year 3",
    "Semester_Year 4":     "Year 4",
}
NUM_LABELS = {
    "Study_Hours_per_Day":   "Study Hours / Day",
    "Attendance_Rate":       "Attendance Rate",
    "Assignment_Delay_Days": "Assignment Delay",
    "Travel_Time_Minutes":   "Travel Time",
    "Stress_Index":          "Stress Index",
    "GPA":                   "GPA",
}

# ========================
# Model loader
# ========================
@st.cache_resource
def load_model():
    path = "logistic_regression_model.pkl"
    if not os.path.exists(path):
        return None, f"Model file not found: `{path}`"
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()
if model_error:
    st.error(f"Model could not be loaded: {model_error}")
    st.stop()

def get_all_feature_names():
    prep = model.named_steps["preprocessor"]
    ohe  = prep.named_transformers_["cat"].named_steps["onehot"]
    return NUMERIC_FEATURES + ohe.get_feature_names_out(CAT_FEATURES).tolist()

def friendly(raw):
    return OHE_LABELS.get(raw, NUM_LABELS.get(raw, raw.replace("_", " ")))

def classify_risk(p):
    if p < 0.35:   return "Low Risk",    "risk-low"
    if p < 0.65:   return "Medium Risk", "risk-med"
    return              "High Risk",   "risk-hi"

def compute_contributions(df_input):
    prep   = model.named_steps["preprocessor"]
    lr     = model.named_steps["model"]
    X_t    = prep.transform(df_input)
    coefs  = lr.coef_[0]
    contribs = X_t[0] * coefs
    names = get_all_feature_names()
    out = pd.DataFrame({
        "feature": names,
        "contribution": contribs,
        "abs": np.abs(contribs),
    }).sort_values("abs", ascending=False).head(10)
    total = out["abs"].sum()
    out["pct"] = (out["abs"] / total * 100).round(1)
    out["dir"] = out["contribution"].apply(lambda x: "+" if x > 0 else "−")
    out["label"] = out["feature"].apply(friendly)
    return out.reset_index(drop=True)

# ========================
# Header
# ========================
st.markdown("""
<div class="app-masthead">
  <div class="masthead-left">
    <div class="masthead-eyebrow">Academic Risk Assessment</div>
    <div class="masthead-title">Student Dropout<br><em>Predictor</em></div>
  </div>
  <div class="masthead-right">
    <div class="masthead-sub">
      Logistic regression model trained on behavioural and academic data.
      Results are probabilistic estimates to support advising.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ========================
# INPUT FORM (Stress & Semester sejajar)
# ========================
st.markdown('<div class="sec-label">Student Profile</div>', unsafe_allow_html=True)

# Baris 1: GPA, Attendance, Study Hours (kiri) vs Delay, Travel, Internet (kanan)
col_left, col_right = st.columns(2, gap="large")

with col_left:
    gpa = st.slider("GPA", 0.0, 4.0, 2.8, 0.05)
    attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5)
    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 3.0, 0.25)

with col_right:
    delay_days = st.slider("Assignment Delay (Days)", 0, 30, 3, 1)
    travel = st.slider("Travel Time (Minutes)", 0, 180, 30, 5)
    internet = st.selectbox("Internet Access", ["Yes", "No"])

# Baris 2: Stress (kiri) dan Semester (kanan) – SEJAJAR HORIZONTAL
stress_col, semester_col = st.columns(2, gap="large")
with stress_col:
    stress = st.slider("Stress Index", 0.0, 10.0, 5.0, 0.1)
with semester_col:
    semester = st.selectbox("Semester", ["Year 1","Year 2","Year 3","Year 4"])

# Baris 3: Part-Time Job dan Scholarship (dua kolom)
pt_col, sch_col = st.columns(2, gap="large")
with pt_col:
    part_time = st.selectbox("Part-Time Job", ["No", "Yes"])
with sch_col:
    scholarship = st.selectbox("Scholarship", ["No", "Yes"])

# Tombol prediksi di tengah
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("Run Prediction", use_container_width=True)

# ========================
# HASIL (ditampilkan setelah predict)
# ========================
if predict:
    df_in = pd.DataFrame([{
        "Study_Hours_per_Day":   float(study_hours),
        "Attendance_Rate":       float(attendance),
        "Assignment_Delay_Days": int(delay_days),
        "Travel_Time_Minutes":   float(travel),
        "Stress_Index":          float(stress),
        "GPA":                   float(gpa),
        "Internet_Access":       internet,
        "Part_Time_Job":         part_time,
        "Scholarship":           scholarship,
        "Semester":              semester,
    }])[INPUT_COLS]

    proba = model.predict_proba(df_in)[0][1]
    label, cls = classify_risk(proba)
    contrib_df = compute_contributions(df_in)

    st.markdown('<div class="sec-label">Prediction Result</div>', unsafe_allow_html=True)
    
    # Gunakan columns untuk hasil (probabilitas dan risk badge)
    colA, colB = st.columns([2, 1], gap="medium")
    with colA:
        st.markdown(f"""
        <div class="result-prob">
          <div style="font-size:0.65rem; letter-spacing:0.15em; color:var(--ink3); margin-bottom:0.2rem;">Dropout Probability</div>
          <div class="prob-number">{proba*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown(f"""
        <div class="result-prob" style="text-align:center;">
          <div style="font-size:0.65rem; letter-spacing:0.15em; color:var(--ink3); margin-bottom:0.2rem;">Risk Level</div>
          <div class="risk-badge {cls}" style="margin-top:0.2rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabel faktor kontribusi
    st.markdown("""
    <div style="margin-top:1.2rem;">
      <div style="font-size:0.65rem; font-weight:600; letter-spacing:0.15em; color:var(--ink3); margin-bottom:0.8rem;">TOP 10 INFLUENCING FACTORS</div>
    </div>
    """, unsafe_allow_html=True)

    for _, row in contrib_df.iterrows():
        bar_cls = "pos-fill" if row["dir"] == "+" else "neg-fill"
        dir_cls = "pos-dir" if row["dir"] == "+" else "neg-dir"
        st.markdown(f"""
        <div class="factor-row">
          <span class="factor-name">{row['label']}</span>
          <div class="factor-bar">
            <div class="factor-bar-fill {bar_cls}" style="width:{row['pct']}%"></div>
          </div>
          <span class="factor-pct">{row['pct']}%</span>
          <span class="factor-dir {dir_cls}">{row['dir']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="legend">
      <div><span class="legend-dot" style="background:var(--pos);"></span> Increases dropout risk</div>
      <div><span class="legend-dot" style="background:var(--neg);"></span> Decreases dropout risk</div>
    </div>
    """, unsafe_allow_html=True)
else:
    pass
