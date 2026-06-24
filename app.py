import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="🫀",
    layout="centered",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .header p {
        color: #6b7280;
        font-size: 1rem;
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.8rem 0 0.8rem 0;
    }

    .card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* Result cards */
    .result-low {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-high {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-label-low  { color: #059669; }
    .result-label-high { color: #e11d48; }
    .result-prob {
        font-size: 1rem;
        color: #374151;
        margin-bottom: 1.2rem;
    }
    .result-note {
        font-size: 0.82rem;
        color: #6b7280;
        font-style: italic;
    }

    /* Gauge bar */
    .gauge-wrap { margin: 1rem 0 0.5rem 0; }
    .gauge-track {
        background: #e5e7eb;
        border-radius: 99px;
        height: 14px;
        overflow: hidden;
        position: relative;
    }
    .gauge-fill-low  { background: linear-gradient(90deg, #34d399, #10b981); border-radius: 99px; height: 100%; transition: width 0.6s ease; }
    .gauge-fill-high { background: linear-gradient(90deg, #fb923c, #f43f5e); border-radius: 99px; height: 100%; transition: width 0.6s ease; }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Factor pills */
    .pill-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.5rem; }
    .pill-risk {
        background: #fee2e2; color: #b91c1c;
        border-radius: 99px; padding: 4px 12px;
        font-size: 0.78rem; font-weight: 500;
    }
    .pill-safe {
        background: #d1fae5; color: #065f46;
        border-radius: 99px; padding: 4px 12px;
        font-size: 0.78rem; font-weight: 500;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
        cursor: pointer;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model & scaler
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ `model.pkl` or `scaler.pkl` not found. Make sure both files are in the same folder as `app.py`.")
    st.stop()

COLS_TO_SCALE = [
    'Age', 'Cholesterol', 'Systolic', 'Diastolic', 'Heart Rate',
    'Exercise Hours Per Week', 'Sedentary Hours Per Day',
    'Income', 'BMI', 'Triglycerides'
]

FEATURE_COLS = [
    'Age', 'Sex', 'Cholesterol', 'Systolic', 'Diastolic', 'Heart Rate',
    'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption',
    'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems', 'Medication Use',
    'Stress Level', 'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
    'Physical Activity Days Per Week', 'Sleep Hours Per Day'
]

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🫀 Heart Attack Risk Predictor</h1>
    <p>Fill in the patient's health profile below to assess their risk.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────

# — Demographics —
st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 100, 50)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    income = c3.number_input("Income (USD/yr)", 0, 500000, 50000, step=5000)

# — Clinical Measurements —
st.markdown('<div class="section-title">Clinical Measurements</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3 = st.columns(3)
    cholesterol  = c1.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    systolic     = c2.number_input("Systolic BP (mmHg)", 80, 200, 120)
    diastolic    = c3.number_input("Diastolic BP (mmHg)", 50, 130, 80)
    c1, c2, c3 = st.columns(3)
    heart_rate   = c1.number_input("Heart Rate (bpm)", 40, 200, 75)
    bmi          = c2.number_input("BMI", 10.0, 60.0, 25.0)
    triglycerides = c3.number_input("Triglycerides (mg/dL)", 50, 800, 150)

# — Medical History —
st.markdown('<div class="section-title">Medical History</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    diabetes      = c1.selectbox("Diabetes", ["No", "Yes"])
    family_hist   = c2.selectbox("Family History", ["No", "Yes"])
    prev_heart    = c3.selectbox("Previous Heart Problems", ["No", "Yes"])
    medication    = c4.selectbox("Medication Use", ["No", "Yes"])

# — Lifestyle —
st.markdown('<div class="section-title">Lifestyle</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    smoking  = c1.selectbox("Smoking", ["No", "Yes"])
    obesity  = c2.selectbox("Obesity", ["No", "Yes"])
    alcohol  = c3.selectbox("Alcohol Consumption", ["No", "Yes"])
    diet     = c4.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    stress       = c1.slider("Stress Level", 1, 10, 5)
    sleep_hours  = c2.slider("Sleep Hours/Day", 4, 12, 7)
    exercise_hrs = c3.number_input("Exercise Hrs/Week", 0.0, 20.0, 3.0)
    sedentary    = c4.number_input("Sedentary Hrs/Day", 0.0, 24.0, 6.0)

with st.container():
    c1, c2 = st.columns(2)
    phys_days = c1.slider("Physical Activity Days/Week", 0, 7, 3)

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
if st.button("Assess Heart Attack Risk"):

    # Encode categoricals
    sex_val     = 1 if sex == "Male" else 0
    diet_val    = {"Healthy": 2, "Average": 1, "Unhealthy": 0}[diet]
    yn          = {"No": 0, "Yes": 1}

    patient = pd.DataFrame([{
        'Age':                             age,
        'Sex':                             sex_val,
        'Cholesterol':                     cholesterol,
        'Systolic':                        systolic,
        'Diastolic':                       diastolic,
        'Heart Rate':                      heart_rate,
        'Diabetes':                        yn[diabetes],
        'Family History':                  yn[family_hist],
        'Smoking':                         yn[smoking],
        'Obesity':                         yn[obesity],
        'Alcohol Consumption':             yn[alcohol],
        'Exercise Hours Per Week':         exercise_hrs,
        'Diet':                            diet_val,
        'Previous Heart Problems':         yn[prev_heart],
        'Medication Use':                  yn[medication],
        'Stress Level':                    stress,
        'Sedentary Hours Per Day':         sedentary,
        'Income':                          income,
        'BMI':                             bmi,
        'Triglycerides':                   triglycerides,
        'Physical Activity Days Per Week': phys_days,
        'Sleep Hours Per Day':             sleep_hours,
    }])

    patient = patient[FEATURE_COLS]
    patient_scaled = patient.copy()
    patient_scaled[COLS_TO_SCALE] = scaler.transform(patient[COLS_TO_SCALE])

    proba = model.predict_proba(patient_scaled)[0][1]
    pred  = int(proba >= 0.5)
    pct   = round(proba * 100, 1)

    st.markdown("---")

    # Result card
    if pred == 0:
        st.markdown(f"""
        <div class="result-low">
            <div class="result-icon">✅</div>
            <div class="result-label result-label-low">Low Risk</div>
            <div class="result-prob">Estimated probability of heart attack risk: <strong>{pct}%</strong></div>
            <div class="gauge-wrap">
                <div class="gauge-track">
                    <div class="gauge-fill-low" style="width:{pct}%"></div>
                </div>
                <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
            </div>
            <div class="result-note">This patient shows a low risk profile based on the provided parameters.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-high">
            <div class="result-icon">⚠️</div>
            <div class="result-label result-label-high">High Risk</div>
            <div class="result-prob">Estimated probability of heart attack risk: <strong>{pct}%</strong></div>
            <div class="gauge-wrap">
                <div class="gauge-track">
                    <div class="gauge-fill-high" style="width:{pct}%"></div>
                </div>
                <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
            </div>
            <div class="result-note">This patient shows a high risk profile. Please consult a medical professional.</div>
        </div>
        """, unsafe_allow_html=True)

    # Contributing factors
    st.markdown("---")
    st.markdown('<div class="section-title">Contributing Factors</div>', unsafe_allow_html=True)

    risk_factors = []
    safe_factors = []

    if yn[smoking]:        risk_factors.append("🚬 Smoking")
    if yn[diabetes]:       risk_factors.append("🩸 Diabetes")
    if yn[family_hist]:    risk_factors.append("🧬 Family History")
    if yn[prev_heart]:     risk_factors.append("💔 Previous Heart Problems")
    if yn[obesity]:        risk_factors.append("⚖️ Obesity")
    if diet == "Unhealthy": risk_factors.append("🍔 Unhealthy Diet")
    if stress >= 7:        risk_factors.append(f"😰 High Stress (Level {stress})")
    if sedentary >= 8:     risk_factors.append(f"🛋️ High Sedentary Hours")
    if cholesterol >= 240: risk_factors.append(f"📈 High Cholesterol ({cholesterol})")
    if bmi >= 30:          risk_factors.append(f"📊 High BMI ({bmi})")

    if not yn[smoking]:       safe_factors.append("✅ Non-smoker")
    if exercise_hrs >= 3:     safe_factors.append(f"🏃 Active ({exercise_hrs}h/week)")
    if diet == "Healthy":     safe_factors.append("🥗 Healthy Diet")
    if stress <= 4:           safe_factors.append(f"😌 Low Stress (Level {stress})")
    if bmi < 25:              safe_factors.append(f"✅ Healthy BMI ({bmi})")
    if not yn[diabetes]:      safe_factors.append("✅ No Diabetes")

    col_r, col_s = st.columns(2)
    with col_r:
        st.markdown("**Risk Factors Present**")
        if risk_factors:
            pills = "".join([f'<span class="pill-risk">{f}</span>' for f in risk_factors])
            st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown("_None identified_")

    with col_s:
        st.markdown("**Protective Factors**")
        if safe_factors:
            pills = "".join([f'<span class="pill-safe">{f}</span>' for f in safe_factors])
            st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown("_None identified_")

    st.markdown("""
    <br>
    <p style='font-size:0.78rem; color:#9ca3af; text-align:center;'>
    ⚕️ This tool is for educational purposes only and does not constitute medical advice.
    Always consult a qualified healthcare professional.
    </p>
    """, unsafe_allow_html=True)
