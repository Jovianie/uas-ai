import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F5F0E8;
  }

  .stApp { background-color: #F5F0E8; }

  /* ── Header ── */
  .app-header {
    background: linear-gradient(135deg, #6B5B47 0%, #8B7355 50%, #A0916D 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(107,91,71,0.25);
  }
  .app-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }
  .app-header p {
    font-size: 1rem;
    opacity: 0.88;
    margin: 0;
    font-weight: 300;
  }

  /* ── Section titles ── */
  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #4A3728;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #D4B896;
  }

  /* ── Cards ── */
  .card {
    background: #FFFDF9;
    border-radius: 16px;
    padding: 1.6rem;
    box-shadow: 0 2px 12px rgba(107,91,71,0.10);
    border: 1px solid #EDE4D5;
    transition: box-shadow 0.2s;
  }
  .card:hover { box-shadow: 0 6px 24px rgba(107,91,71,0.16); }

  /* ── Metric Card ── */
  .metric-card {
    background: linear-gradient(135deg, #6B5B47, #8B7355);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 24px rgba(107,91,71,0.30);
  }
  .metric-card .label {
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    opacity: 0.80;
    margin-bottom: 0.5rem;
  }
  .metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1;
  }

  /* ── Risk badges ── */
  .risk-low {
    background: linear-gradient(135deg, #5B8A6B, #7AAD8A);
    color: white;
    border-radius: 50px;
    padding: 0.65rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    display: inline-block;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 12px rgba(91,138,107,0.35);
  }
  .risk-medium {
    background: linear-gradient(135deg, #C49A3C, #E8BA5A);
    color: white;
    border-radius: 50px;
    padding: 0.65rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    display: inline-block;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 12px rgba(196,154,60,0.35);
  }
  .risk-high {
    background: linear-gradient(135deg, #B85450, #D4706D);
    color: white;
    border-radius: 50px;
    padding: 0.65rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    display: inline-block;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 12px rgba(184,84,80,0.35);
  }

  /* ── Factor bars ── */
  .factor-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.75rem;
  }
  .factor-rank {
    font-size: 0.75rem;
    color: #8B7355;
    font-weight: 600;
    width: 22px;
    text-align: center;
  }
  .factor-name {
    font-size: 0.88rem;
    font-weight: 500;
    color: #4A3728;
    width: 210px;
    flex-shrink: 0;
  }
  .factor-bar-bg {
    flex: 1;
    background: #EDE4D5;
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
  }
  .factor-bar-fill-pos {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #B85450, #D4706D);
  }
  .factor-bar-fill-neg {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #5B8A6B, #7AAD8A);
  }
  .factor-pct {
    font-size: 0.82rem;
    font-weight: 600;
    color: #6B5B47;
    width: 48px;
    text-align: right;
  }
  .factor-dir {
    font-size: 0.80rem;
    font-weight: 700;
    width: 18px;
    text-align: center;
  }

  /* ── Intervention cards ── */
  .intervention-card {
    background: #FFFDF9;
    border-left: 4px solid #8B7355;
    border-radius: 0 12px 12px 0;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 8px rgba(107,91,71,0.08);
  }
  .intervention-card .int-icon { font-size: 1.3rem; margin-right: 0.5rem; }
  .intervention-card .int-title {
    font-weight: 600;
    color: #4A3728;
    font-size: 0.95rem;
  }
  .intervention-card .int-text {
    color: #6B5B47;
    font-size: 0.88rem;
    margin-top: 0.35rem;
    line-height: 1.55;
  }

  /* ── Ethical notice ── */
  .ethical-notice {
    background: linear-gradient(135deg, #EDE4D5, #F5EDD9);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    border: 1px solid #D4B896;
    font-size: 0.88rem;
    color: #6B5B47;
    line-height: 1.6;
  }
  .ethical-notice strong { color: #4A3728; }

  /* ── Form labels ── */
  .stSlider label, .stSelectbox label, .stNumberInput label {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #4A3728 !important;
  }

  /* ── Predict button ── */
  .stButton > button {
    background: linear-gradient(135deg, #6B5B47, #8B7355);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 0.75rem 3rem;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 16px rgba(107,91,71,0.30);
    transition: all 0.2s;
    width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #5A4A38, #7A6347);
    box-shadow: 0 6px 20px rgba(107,91,71,0.40);
    transform: translateY(-1px);
  }

  /* ── Divider ── */
  hr { border-color: #EDE4D5 !important; }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "logistic_regression_model.pkl"
    if not os.path.exists(model_path):
        return None, f"Model file not found at `{model_path}`. Please place the PKL file in the app directory."
    try:
        m = joblib.load(model_path)
        return m, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


model, model_error = load_model()

# ── Feature metadata ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Study_Hours_per_Day",
    "Attendance_Rate",
    "Assignment_Delay_Days",
    "Travel_Time_Minutes",
    "Stress_Index",
    "GPA",
]
CAT_FEATURES = ["Internet_Access", "Part_Time_Job", "Scholarship", "Semester"]
INPUT_COLS = NUMERIC_FEATURES + CAT_FEATURES  # expected pipeline order

FEATURE_LABELS = {
    "Study_Hours_per_Day": "Study Hours per Day",
    "Attendance_Rate": "Attendance Rate (%)",
    "Assignment_Delay_Days": "Assignment Delay (Days)",
    "Travel_Time_Minutes": "Travel Time (Minutes)",
    "Stress_Index": "Stress Index",
    "GPA": "GPA",
    "Internet_Access": "Internet Access",
    "Part_Time_Job": "Part-Time Job",
    "Scholarship": "Scholarship",
    "Semester": "Current Semester",
}

# OHE expanded names → human-readable label
OHE_LABELS = {
    "Internet_Access_No": "No Internet Access",
    "Internet_Access_Yes": "Has Internet Access",
    "Part_Time_Job_No": "No Part-Time Job",
    "Part_Time_Job_Yes": "Has Part-Time Job",
    "Scholarship_No": "No Scholarship",
    "Scholarship_Yes": "Has Scholarship",
    "Semester_Year 1": "Semester: Year 1",
    "Semester_Year 2": "Semester: Year 2",
    "Semester_Year 3": "Semester: Year 3",
    "Semester_Year 4": "Semester: Year 4",
}


def get_all_feature_names():
    preprocessor = model.named_steps["preprocessor"]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    ohe_names = onehot.get_feature_names_out(CAT_FEATURES).tolist()
    return NUMERIC_FEATURES + ohe_names


def friendly_name(raw):
    if raw in OHE_LABELS:
        return OHE_LABELS[raw]
    return FEATURE_LABELS.get(raw, raw.replace("_", " "))


# ── Risk classification ─────────────────────────────────────────────────────────
def classify_risk(prob):
    if prob < 0.35:
        return "Low Risk", "risk-low", "🟢"
    elif prob < 0.65:
        return "Medium Risk", "risk-medium", "🟡"
    else:
        return "High Risk", "risk-high", "🔴"


# ── Feature contributions (LR coefficients × transformed input) ─────────────────
def compute_contributions(input_df):
    preprocessor = model.named_steps["preprocessor"]
    lr = model.named_steps["model"]

    X_transformed = preprocessor.transform(input_df)
    coefs = lr.coef_[0]
    contributions = X_transformed[0] * coefs

    feature_names = get_all_feature_names()
    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "raw_contribution": contributions,
    })
    contrib_df["abs"] = contrib_df["raw_contribution"].abs()
    contrib_df = contrib_df.sort_values("abs", ascending=False).head(10)

    total = contrib_df["abs"].sum()
    contrib_df["pct"] = (contrib_df["abs"] / total * 100).round(1)
    contrib_df["direction"] = contrib_df["raw_contribution"].apply(
        lambda x: "+" if x > 0 else "-"
    )
    contrib_df["label"] = contrib_df["feature"].apply(friendly_name)
    return contrib_df.reset_index(drop=True)


# ── Ethical intervention generator ─────────────────────────────────────────────
def generate_interventions(contrib_df, prob):
    interventions = []
    top_features = contrib_df[contrib_df["direction"] == "+"]["feature"].tolist()

    def has(kw):
        return any(kw.lower() in f.lower() for f in top_features)

    if has("Attendance"):
        interventions.append({
            "icon": "📅",
            "title": "Improve Class Attendance",
            "text": (
                "Your attendance rate is a key driver in this prediction. "
                "Try setting calendar reminders for each class, connecting with a study "
                "partner for accountability, and speaking with your academic advisor if "
                "circumstances make attendance difficult."
            ),
        })
    if has("Stress"):
        interventions.append({
            "icon": "🧘",
            "title": "Manage Stress Levels",
            "text": (
                "Elevated stress is significantly influencing your risk score. "
                "Consider exploring your campus counselling services, practising "
                "mindfulness or breathing exercises, and ensuring you schedule "
                "regular breaks to avoid burnout."
            ),
        })
    if has("Study_Hours"):
        interventions.append({
            "icon": "📚",
            "title": "Build Consistent Study Habits",
            "text": (
                "Low or irregular study hours are contributing to your risk. "
                "Set a daily study schedule, break large tasks into 25-minute "
                "Pomodoro sessions, and find a quiet, dedicated study environment "
                "on or off campus."
            ),
        })
    if has("Assignment_Delay"):
        interventions.append({
            "icon": "⏰",
            "title": "Reduce Assignment Delays",
            "text": (
                "Frequent late submissions are flagged as a risk factor. "
                "Use a planner or task-management app to track deadlines, break "
                "assignments into smaller milestones, and communicate proactively "
                "with instructors when you foresee delays."
            ),
        })
    if has("GPA"):
        interventions.append({
            "icon": "🎯",
            "title": "Seek Academic Support",
            "text": (
                "Academic performance is playing a role in this prediction. "
                "Connect with your faculty's tutoring centre, form study groups "
                "with classmates, and attend office hours early — before "
                "grades become critical."
            ),
        })
    if has("Part_Time_Job"):
        interventions.append({
            "icon": "💼",
            "title": "Balance Work and Study",
            "text": (
                "Balancing part-time employment and academic demands can be "
                "challenging. Explore campus work-study programmes with "
                "flexible hours, discuss your schedule with your employer, "
                "and connect with your institution's financial aid office."
            ),
        })
    if has("Travel_Time"):
        interventions.append({
            "icon": "🚌",
            "title": "Optimise Your Commute",
            "text": (
                "Long daily travel time is a contributing factor. Look into "
                "on-campus housing options, carpooling with classmates, or "
                "using commute time productively with recorded lectures "
                "or study podcasts."
            ),
        })
    if has("Internet") or has("Scholarship"):
        interventions.append({
            "icon": "💡",
            "title": "Explore Resource & Financial Support",
            "text": (
                "Access to resources and financial stability significantly "
                "affect academic success. Visit your student services office "
                "to learn about scholarships, emergency bursaries, digital "
                "access programmes, and other support available to you."
            ),
        })

    # Generic fallback
    if not interventions:
        interventions.append({
            "icon": "🤝",
            "title": "Connect with Academic Support",
            "text": (
                "Reaching out to an academic advisor or student support "
                "coordinator is always a positive first step. They can "
                "help you build a personalised plan to stay on track."
            ),
        })

    return interventions[:4]


# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
  <h1>🎓 Student Dropout Predictor</h1>
  <p>A machine-learning tool to identify at-risk students early and guide timely, compassionate intervention.</p>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.error(f"⚠️ {model_error}")
    st.stop()

# ── Two-column layout ──────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.15], gap="large")

with left_col:
    st.markdown('<div class="section-title">📋 Student Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("**Academic Performance**")
    gpa = st.slider("GPA", min_value=0.0, max_value=4.0, value=2.8, step=0.05,
                    help="Cumulative Grade Point Average (0.0 – 4.0)")
    attendance = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0,
                           value=75.0, step=0.5,
                           help="Percentage of classes attended")

    st.markdown("---")
    st.markdown("**Study Habits**")
    study_hours = st.slider("Study Hours per Day", min_value=0.0, max_value=12.0,
                             value=3.0, step=0.25,
                             help="Average daily self-study hours")
    delay_days = st.slider("Assignment Delay (Days)", min_value=0, max_value=30,
                            value=3, step=1,
                            help="Average days late for submitting assignments")

    st.markdown("---")
    st.markdown("**Wellbeing & Lifestyle**")
    stress = st.slider("Stress Index", min_value=0.0, max_value=10.0, value=5.0,
                        step=0.1, help="Self-reported stress level (0 = none, 10 = extreme)")
    travel = st.slider("Travel Time (Minutes)", min_value=0, max_value=180,
                        value=30, step=5,
                        help="Daily one-way commute time in minutes")

    st.markdown("---")
    st.markdown("**Background & Resources**")

    col_a, col_b = st.columns(2)
    with col_a:
        internet = st.selectbox("Internet Access", ["Yes", "No"], index=0)
        scholarship = st.selectbox("Scholarship", ["No", "Yes"], index=0)
    with col_b:
        part_time = st.selectbox("Part-Time Job", ["No", "Yes"], index=0)
        semester = st.selectbox("Current Semester",
                                ["Year 1", "Year 2", "Year 3", "Year 4"],
                                index=0)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🔍 Predict Dropout Risk", use_container_width=True)

# ── Right column: results ──────────────────────────────────────────────────────
with right_col:
    if not predict_clicked:
        st.markdown("""
        <div class="card" style="text-align:center; padding: 3rem 2rem; color: #8B7355;">
          <div style="font-size:3.5rem; margin-bottom:1rem;">🎓</div>
          <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                      font-weight:600; color:#4A3728; margin-bottom:0.6rem;">
            Ready to Analyse
          </div>
          <div style="font-size:0.92rem; line-height:1.6;">
            Fill in the student information on the left<br>and click <strong>Predict Dropout Risk</strong><br>
            to see the results.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Build input DataFrame ──────────────────────────────────────────────
        try:
            input_data = pd.DataFrame([{
                "Study_Hours_per_Day": float(study_hours),
                "Attendance_Rate": float(attendance),
                "Assignment_Delay_Days": int(delay_days),
                "Travel_Time_Minutes": float(travel),
                "Stress_Index": float(stress),
                "GPA": float(gpa),
                "Internet_Access": internet,
                "Part_Time_Job": part_time,
                "Scholarship": scholarship,
                "Semester": semester,
            }])
            input_data = input_data[INPUT_COLS]

            proba = model.predict_proba(input_data)[0][1]
            risk_label, risk_class, risk_icon = classify_risk(proba)
            contrib_df = compute_contributions(input_data)
            interventions = generate_interventions(contrib_df, proba)

            # ── 1. Probability metric card ─────────────────────────────────────
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Student Dropout Probability</div>
              <div class="value">{proba*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 2. Risk classification ─────────────────────────────────────────
            st.markdown('<div class="section-title">⚡ Risk Classification</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="text-align:center; padding:1.4rem;">'
                        f'<span class="{risk_class}">{risk_icon} {risk_label}</span>'
                        f'<div style="margin-top:0.9rem; font-size:0.84rem; color:#8B7355;">'
                        f'Threshold: Low &lt; 35% · Medium 35–65% · High &gt; 65%'
                        f'</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 3. Top influencing factors ─────────────────────────────────────
            st.markdown('<div class="section-title">📊 Top 10 Influencing Factors</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)

            rows_html = ""
            for idx, row in contrib_df.iterrows():
                bar_class = "factor-bar-fill-pos" if row["direction"] == "+" else "factor-bar-fill-neg"
                dir_color = "#B85450" if row["direction"] == "+" else "#5B8A6B"
                rows_html += f"""
                <div class="factor-row">
                  <span class="factor-rank">#{idx+1}</span>
                  <span class="factor-name">{row['label']}</span>
                  <div class="factor-bar-bg">
                    <div class="{bar_class}" style="width:{row['pct']}%"></div>
                  </div>
                  <span class="factor-pct">{row['pct']}%</span>
                  <span class="factor-dir" style="color:{dir_color}">{row['direction']}</span>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.78rem;color:#8B7355;margin-top:0.5rem;">'
                '🔴 <strong>+</strong> increases dropout risk &nbsp;|&nbsp; '
                '🟢 <strong>−</strong> decreases dropout risk</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}. Please check your inputs and try again.")
            st.stop()

# ── Ethical intervention section (full-width) ───────────────────────────────────
if predict_clicked and model_error is None:
    try:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">💚 Personalised Recommendations</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="ethical-notice">
          <strong>⚠️ Important Notice:</strong> This prediction is <em>probabilistic</em>, not deterministic.
          It is a supportive tool — not a verdict. A high probability score does <em>not</em> mean a student
          will drop out. Many students overcome risk factors with the right support.
          These recommendations are designed to <strong>empower</strong>, not stigmatise.
        </div>
        """, unsafe_allow_html=True)

        cards_html = ""
        for iv in interventions:
            cards_html += f"""
            <div class="intervention-card">
              <div>
                <span class="int-icon">{iv['icon']}</span>
                <span class="int-title">{iv['title']}</span>
              </div>
              <div class="int-text">{iv['text']}</div>
            </div>
            """
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.82rem; color:#8B7355; margin-top:1rem; text-align:center;">
          Recommendations are generated dynamically based on the dominant risk factors
          identified for this student profile.
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        pass
