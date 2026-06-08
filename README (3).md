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
A large percentage card showing the model's estimated probability that the student will drop out. Example: `72.84%`

#### 2. Risk Classification
A colour-coded badge indicating the risk tier:
| Badge | Range | Meaning |
|---|---|---|
| 🟢 Low Risk | < 35% | Student is unlikely to drop out |
| 🟡 Medium Risk | 35% – 65% | Student has moderate risk; monitoring recommended |
| 🔴 High Risk | > 65% | Student is at high risk; immediate support advised |

#### 3. Top 10 Influencing Factors
A ranked bar chart listing the ten features that most influenced this particular prediction, showing:
- **Factor name** — The variable driving the prediction
- **Influence (%)** — Relative contribution normalised to 100%
- **Direction** — `+` means it increases dropout risk; `−` means it decreases risk

---

### Step 4 — Review Personalised Recommendations

Below the results, the **Personalised Recommendations** section provides dynamic, evidence-informed suggestions tailored to the dominant risk factors identified for the student. Categories include:

- 📅 Attendance improvement strategies
- 🧘 Stress management resources
- 📚 Study habit building
- ⏰ Assignment and time management tips
- 🎯 Academic support pathways
- 💼 Work-study balance guidance
- 🚌 Commute optimisation
- 💡 Financial and resource support

> **Note:** Recommendations change automatically based on which factors most influence the prediction for each student.

---

## Important Ethical Notice

This tool provides **probabilistic estimates**, not deterministic outcomes.

- A high risk score does **not** mean a student will drop out.
- Many students with high risk scores succeed with the right support.
- This tool is intended to guide early, compassionate intervention — not to label or stigmatise students.
- Predictions should be used alongside professional academic advising and student support services.

---

## Local Setup

```bash
# Clone or download the project
cd student_dropout_app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Make sure `logistic_regression_model.pkl` is in the **same directory** as `app.py`.

---

## Deployment on Streamlit Cloud

1. Push this project to a GitHub repository (include `app.py`, `requirements.txt`, and `logistic_regression_model.pkl`).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in.
3. Click **New app** → select your repository → set the main file path to `app.py`.
4. Click **Deploy**.

The app will be live within a few minutes.

---

## Project Structure

```
student_dropout_app/
│
├── app.py                          # Main Streamlit application
├── logistic_regression_model.pkl   # Trained Logistic Regression pipeline
├── requirements.txt                # Pinned Python dependencies
├── README.md                       # This usage guide
└── .gitignore                      # Git ignore rules
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Pipeline | `ColumnTransformer` → `LogisticRegression` |
| Numeric preprocessing | `SimpleImputer(median)` → `StandardScaler` |
| Categorical preprocessing | `SimpleImputer(most_frequent)` → `OneHotEncoder` |
| Input features | 10 (6 numeric, 4 categorical) |
| Output | Dropout probability (class 1) |
| scikit-learn version | 1.6.1 |
