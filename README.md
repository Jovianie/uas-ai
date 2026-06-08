# 🎓 Student Dropout Predictor

A production-ready Streamlit web application that predicts student dropout risk using a Logistic Regression model trained on student behavioural and academic data.

---

## How to Use the App

### Step 1 — Fill In Student Information

On the left panel, enter the student's details across four sections:

**Academic Performance**
- **GPA** — Cumulative Grade Point Average on a 0.0–4.0 scale. Use the slider to set the value.
- **Attendance Rate (%)** — Percentage of total classes attended. A rate below 75% is generally considered at-risk.

**Study Habits**
- **Study Hours per Day** — Average number of hours the student spends studying outside class per day.
- **Assignment Delay (Days)** — Average number of days the student submits assignments after the deadline. 0 means on time.

**Wellbeing & Lifestyle**
- **Stress Index** — Self-reported stress level on a scale of 0 (none) to 10 (extreme).
- **Travel Time (Minutes)** — One-way daily commute time in minutes.

**Background & Resources**
- **Internet Access** — Whether the student has reliable internet access at home (Yes / No).
- **Part-Time Job** — Whether the student is currently working part-time (Yes / No).
- **Scholarship** — Whether the student receives a scholarship (Yes / No).
- **Current Semester** — The student's current academic year (Year 1 through Year 4).

---

### Step 2 — Click "Predict Dropout Risk"

After filling in all fields, click the **Predict Dropout Risk** button at the bottom of the form.

---

### Step 3 — Read the Results

The right panel will display three result sections:

#### 1. Dropout Probability
A large percentage card showing the model's estimated probability that the student will drop out. Example: `72.8%`

#### 2. Top 10 Influencing Factors
A ranked bar chart listing the ten features that most influenced this particular prediction, showing:
- **Factor name** — The variable driving the prediction
- **Influence (%)** — Relative contribution normalised to 100%
- **Direction** — `+` means it increases dropout risk; `−` means it decreases risk

## Important Ethical Notice

This tool provides **probabilistic estimates**, not deterministic outcomes.

- A high risk score does **not** mean a student will drop out.
- Many students with high risk scores succeed with the right support.
- This tool is intended to guide early, compassionate intervention — not to label or stigmatise students.
- Predictions should be used alongside professional academic advising and student support services.

---
