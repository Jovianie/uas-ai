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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300;1,9..40,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

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
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }

/* ── HEADER ── */
.app-masthead {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  padding: 2.8rem 3rem 2.2rem;
  background: var(--ink);
  border-radius: 6px;
  margin-bottom: 2.2rem;
  position: relative;
  overflow: hidden;
}
.app-masthead::before {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    -45deg,
    transparent,
    transparent 28px,
    rgba(255,255,255,0.025) 28px,
    rgba(255,255,255,0.025) 29px
  );
}
.masthead-left { position: relative; z-index: 1; }
.masthead-eyebrow {
  font-family: var(--sans);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent2);
  margin-bottom: 0.55rem;
}
.masthead-title {
  font-family: var(--serif);
  font-size: 2.5rem;
  font-weight: 400;
  color: #FAF6EF;
  line-height: 1.08;
  letter-spacing: -0.02em;
}
.masthead-title em {
  font-style: italic;
  color: var(--accent2);
}
.masthead-right {
  position: relative;
  z-index: 1;
  text-align: right;
}
.masthead-sub {
  font-family: var(--sans);
  font-size: 0.84rem;
  color: rgba(250,246,239,0.55);
  font-weight: 300;
  max-width: 280px;
  line-height: 1.55;
}

/* ── SECTION LABEL ── */
.sec-label {
  font-family: var(--sans);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink3);
  margin-bottom: 1rem;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--border);
}
.subsec-label {
  font-family: var(--sans);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 1.4rem 0 0.7rem;
}

/* ── PANEL ── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.8rem 2rem;
}
.panel-tight {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label {
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--ink2) !important;
  letter-spacing: 0.01em !important;
}
div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
}
div[data-testid="stSlider"] .st-emotion-cache-1dp5vir,
div[data-testid="stSlider"] [class*="StyledSlider"] {
  color: var(--accent) !important;
}
.stSelectbox > div > div {
  border-color: var(--border) !important;
  border-radius: 3px !important;
  font-family: var(--sans) !important;
  font-size: 0.84rem !important;
  background: var(--surface) !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  background: var(--ink) !important;
  color: #FAF6EF !important;
  border: none !important;
  border-radius: 3px !important;
  padding: 0.8rem 2rem !important;
  width: 100% !important;
  transition: background 0.18s !important;
}
.stButton > button:hover {
  background: var(--accent) !important;
}

/* ── PROBABILITY BLOCK ── */
.prob-block {
  background: var(--ink);
  border-radius: 4px;
  padding: 2.4rem 2rem;
  text-align: center;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}
.prob-block::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 3px;
  background: var(--accent2);
}
.prob-eyebrow {
  font-family: var(--sans);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: rgba(250,246,239,0.50);
  margin-bottom: 0.9rem;
}
.prob-value {
  font-family: var(--serif);
  font-size: 4.8rem;
  font-weight: 400;
  color: #FAF6EF;
  line-height: 1;
  letter-spacing: -0.03em;
}
.prob-sub {
  font-family: var(--sans);
  font-size: 0.78rem;
  color: rgba(250,246,239,0.40);
  margin-top: 0.6rem;
  font-weight: 300;
  font-style: italic;
}

/* ── RISK BADGE ── */
.risk-wrap { text-align: center; margin-bottom: 1rem; }
.risk-badge {
  display: inline-block;
  font-family: var(--sans);
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 0.5rem 1.6rem;
  border-radius: 2px;
}
.risk-low  { background: #E8F2EC; color: var(--risk-low);  border: 1px solid #B8D9C4; }
.risk-med  { background: #F5EDD5; color: var(--risk-mid);  border: 1px solid #DFC98A; }
.risk-hi   { background: #F2E4E3; color: var(--risk-hi);   border: 1px solid #D9A8A6; }
.risk-threshold {
  font-family: var(--sans);
  font-size: 0.72rem;
  color: var(--ink3);
  margin-top: 0.55rem;
  font-style: italic;
}

/* ── FACTOR TABLE ── */
.factor-header {
  display: grid;
  grid-template-columns: 22px 1fr 120px 46px 20px;
  gap: 8px;
  align-items: center;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.7rem;
}
.factor-header span {
  font-family: var(--sans);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink3);
}
.factor-row {
  display: grid;
  grid-template-columns: 22px 1fr 120px 46px 20px;
  gap: 8px;
  align-items: center;
  padding: 0.42rem 0;
  border-bottom: 1px solid #EDE6D8;
}
.factor-row:last-child { border-bottom: none; }
.f-rank {
  font-family: var(--sans);
  font-size: 0.68rem;
  color: var(--ink3);
  font-weight: 500;
  text-align: center;
}
.f-name {
  font-family: var(--sans);
  font-size: 0.80rem;
  font-weight: 500;
  color: var(--ink2);
}
.f-bar-bg {
  background: #EDE6D8;
  border-radius: 1px;
  height: 7px;
  overflow: hidden;
}
.f-bar-pos { height: 100%; border-radius: 1px; background: var(--pos); }
.f-bar-neg { height: 100%; border-radius: 1px; background: var(--neg); }
.f-pct {
  font-family: var(--sans);
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--ink);
  text-align: right;
}
.f-dir-pos { font-family: var(--sans); font-size: 0.72rem; font-weight: 700; color: var(--pos); text-align: center; }
.f-dir-neg { font-family: var(--sans); font-size: 0.72rem; font-weight: 700; color: var(--neg); text-align: center; }

/* ── LEGEND ── */
.legend {
  display: flex;
  gap: 1.4rem;
  margin-top: 0.9rem;
  padding-top: 0.7rem;
  border-top: 1px solid var(--border);
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-family: var(--sans);
  font-size: 0.72rem;
  color: var(--ink3);
  font-style: italic;
}
.legend-dot {
  width: 8px; height: 8px;
  border-radius: 1px;
  flex-shrink: 0;
}

/* ── PLACEHOLDER ── */
.placeholder {
  background: var(--surface);
  border: 1px dashed var(--border);
  border-radius: 4px;
  padding: 4rem 2rem;
  text-align: center;
}
.placeholder-title {
  font-family: var(--serif);
  font-size: 1.4rem;
  font-style: italic;
  color: var(--ink3);
  margin-bottom: 0.5rem;
}
.placeholder-sub {
  font-family: var(--sans);
  font-size: 0.82rem;
  color: var(--ink3);
  font-weight: 300;
}

hr { border: none; border-top: 1px solid var(--border) !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
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


# ── Model ──────────────────────────────────────────────────────────────────────
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


def get_all_feature_names():
    prep = model.named_steps["preprocessor"]
    ohe  = prep.named_transformers_["cat"].named_steps["onehot"]
    return NUMERIC_FEATURES + ohe.get_feature_names_out(CAT_FEATURES).tolist()


def friendly(raw):
    if raw in OHE_LABELS: return OHE_LABELS[raw]
    return NUM_LABELS.get(raw, raw.replace("_", " "))


def classify_risk(p):
    if p < 0.35:   return "Low Risk",    "risk-low", "< 35%"
    if p < 0.65:   return "Medium Risk", "risk-med", "35 – 65%"
    return              "High Risk",   "risk-hi",  "> 65%"


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


# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-masthead">
  <div class="masthead-left">
    <div class="masthead-eyebrow">Academic Risk Assessment</div>
    <div class="masthead-title">Student Dropout<br><em>Predictor</em></div>
  </div>
  <div class="masthead-right">
    <div class="masthead-sub">
      A logistic regression model trained on student behavioural and
      academic data. Results are probabilistic estimates intended to
      support — not replace — professional advising.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.error(f"Model could not be loaded: {model_error}")
    st.stop()

# ── Two-column layout ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown('<div class="sec-label">Student Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown('<div class="subsec-label">Academic Performance</div>', unsafe_allow_html=True)
    gpa = st.slider("GPA", 0.0, 4.0, 2.8, 0.05,
                    help="Cumulative Grade Point Average (0.0 – 4.0)")
    attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5)

    st.markdown('<div class="subsec-label">Study Habits</div>', unsafe_allow_html=True)
    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 3.0, 0.25)
    delay_days  = st.slider("Assignment Delay (Days)", 0, 30, 3, 1,
                            help="Average days past deadline")

    st.markdown('<div class="subsec-label">Wellbeing & Lifestyle</div>', unsafe_allow_html=True)
    stress = st.slider("Stress Index", 0.0, 10.0, 5.0, 0.1,
                       help="0 = no stress, 10 = extreme stress")
    travel = st.slider("Travel Time (Minutes)", 0, 180, 30, 5,
                       help="One-way daily commute")

    st.markdown('<div class="subsec-label">Background & Resources</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        internet    = st.selectbox("Internet Access", ["Yes", "No"])
        scholarship = st.selectbox("Scholarship", ["No", "Yes"])
    with cb:
        part_time = st.selectbox("Part-Time Job", ["No", "Yes"])
        semester  = st.selectbox("Semester", ["Year 1","Year 2","Year 3","Year 4"])

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("Run Prediction", use_container_width=True)

with right:
    if not predict:
        st.markdown("""
        <div class="placeholder">
          <div class="placeholder-title">Awaiting input</div>
          <div class="placeholder-sub">Complete the student profile and click Run Prediction.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
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

            proba       = model.predict_proba(df_in)[0][1]
            label, cls, thresh = classify_risk(proba)
            contrib_df  = compute_contributions(df_in)

            # — Probability block
            st.markdown(f"""
            <div class="prob-block">
              <div class="prob-eyebrow">Dropout Probability</div>
              <div class="prob-value">{proba*100:.2f}%</div>
              <div class="prob-sub">Logistic Regression · sklearn pipeline</div>
            </div>
            """, unsafe_allow_html=True)

            # — Risk badge
            st.markdown(f"""
            <div class="risk-wrap">
              <span class="risk-badge {cls}">{label}</span>
              <div class="risk-threshold">Threshold: {thresh}</div>
            </div>
            """, unsafe_allow_html=True)

            # — Factor table
            st.markdown('<div class="sec-label" style="margin-top:1.4rem;">Top 10 Influencing Factors</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            rows_html = """
            <div class="factor-header">
              <span>#</span><span>Factor</span><span>Influence</span>
              <span style="text-align:right">Weight</span><span></span>
            </div>
            """
            for i, row in contrib_df.iterrows():
                bar_cls = "f-bar-pos" if row["dir"] == "+" else "f-bar-neg"
                dir_cls = "f-dir-pos" if row["dir"] == "+" else "f-dir-neg"
                rows_html += f"""
                <div class="factor-row">
                  <span class="f-rank">{i+1}</span>
                  <span class="f-name">{row['label']}</span>
                  <div class="f-bar-bg">
                    <div class="{bar_cls}" style="width:{row['pct']}%"></div>
                  </div>
                  <span class="f-pct">{row['pct']}%</span>
                  <span class="{dir_cls}">{row['dir']}</span>
                </div>
                """
            rows_html += """
            <div class="legend">
              <div class="legend-item">
                <div class="legend-dot" style="background:var(--pos)"></div>
                Increases dropout risk
              </div>
              <div class="legend-item">
                <div class="legend-dot" style="background:var(--neg)"></div>
                Decreases dropout risk
              </div>
            </div>
            """
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # — Disclaimer
            st.markdown("""
            <div style="margin-top:1.2rem; padding:1rem 1.2rem;
                        background:var(--surface); border:1px solid var(--border);
                        border-radius:4px; border-left: 3px solid var(--accent2);">
              <p style="font-family:var(--sans); font-size:0.76rem;
                        color:var(--ink3); line-height:1.65; font-style:italic;">
                This result is a probabilistic estimate, not a deterministic outcome.
                Many students with elevated risk scores succeed with timely support.
                This tool is intended to assist academic advisors — not to label
                or stigmatise individual students.
              </p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
