import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Failure Risk Prediction",
    page_icon="🫀",
    layout="wide"
)

# ---------------- ENHANCED THEME ----------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 1400px;
    }

    .stForm {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    div[data-testid="stVerticalBlock"] > div {
        background: transparent !important;
    }

    div[data-testid="column"] {
        background: transparent !important;
        border: none !important;
    }

    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: white !important;
        border-radius: 8px !important;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }

    div[data-testid="stFormSubmitButton"] {
        margin-top: 1.5rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        color: white;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5);
        margin-bottom: 2.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }

    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        margin: 0 0 1.5rem 0;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
        text-shadow: 0 8px 40px rgba(0, 0, 0, 0.5);
        line-height: 1.2;
        word-spacing: 0.1em;
    }
    .hero-subtitle {
        font-size: 1.4rem;
        margin: 0;
        opacity: 0.95;
        font-weight: 300;
        line-height: 1.7;
        position: relative;
        z-index: 1;
        max-width: 1000px;
        margin: 0 auto;
    }

    .card {
        border-radius: 16px;
        padding: 2rem;
        background: transparent;
        border: none;
        box-shadow: none;
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #667eea;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        color: #667eea;
    }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-5px);
    }

    .result-high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }

    .result-low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }

    .probability-display {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .small-text {
        font-size: 0.9rem;
        color: #888;
        margin-top: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🫀 Heart Failure Predictor")
st.sidebar.markdown(
    """
    ### About This Tool

    This application uses machine learning to assess heart disease risk based on clinical parameters.

    **Technology Stack:**
    - 🤖 ML Model: Support Vector Machine
    - 🔬 Feature Engineering: PCA
    - ⚡ Backend: FastAPI
    - 🎨 Frontend: Streamlit

    ---
    ### How to Use

    1. Enter patient demographic information  
    2. Provide diagnostic measurements  
    3. Click **Analyze Risk Profile**  
    4. Review the risk assessment

    ---
    ⚠️ **Disclaimer:** Educational demo only. Not for clinical use.
    """
)

# ---------------- HERO HEADER ----------------
st.markdown(
    """
    <div class="hero-header">
        <p class="hero-title">🫀 Heart Failure Risk Prediction</p>
        <p class="hero-subtitle">
            Advanced clinical decision support powered by machine learning to estimate cardiovascular disease probability.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- MAIN FORM ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">👤 Demographics & Vital Signs</p>', unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)
        sex = st.selectbox("Sex", ["M", "F"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

    with col2:
        st.markdown('<p class="section-header">🔬 Diagnostic Features</p>', unsafe_allow_html=True)
        chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)

    submitted = st.form_submit_button("🔮 Analyze Risk Profile")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION RESULTS ----------------
if submitted:
    payload = {
        "age": age,
        "sex": sex,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "max_hr": max_hr,
        "oldpeak": oldpeak,
        "chest_pain_type": chest_pain_type,
        "resting_ecg": resting_ecg,
        "exercise_angina": exercise_angina,
        "st_slope": st_slope,
    }

    with st.spinner("🔄 Analyzing patient data..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                pred = result["prediction"]
                prob = result["probability"]

                col1, col2 = st.columns([1, 1])

                with col1:
                    risk_class = "result-high-risk" if pred == 1 else "result-low-risk"
                    risk_text = "High Risk Detected" if pred == 1 else "Low Risk Profile"
                    risk_icon = "⚠️" if pred == 1 else "✅"

                    st.markdown(
                        f"""
                        <div class="result-card {risk_class}">
                            <div class="result-title">{risk_icon} {risk_text}</div>
                            <div class="result-subtitle">Model Classification Result</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    prob_color = "#f5576c" if pred == 1 else "#00f2fe"
                    st.markdown(
                        f"""
                        <div class="result-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);">
                            <div class="result-subtitle" style="color: #888;">Disease Probability</div>
                            <div class="probability-display" style="color: {prob_color};">{prob:.1%}</div>
                            <div class="result-subtitle" style="color: #888;">Confidence Score</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="card-title">📊 Clinical Interpretation</p>', unsafe_allow_html=True)

                if pred == 1:
                    st.markdown(
                        """
                        <div class="info-box">
                        The predictive model indicates an <strong>elevated risk</strong> of cardiovascular disease based on the provided clinical parameters. 
                        This assessment suggests that further diagnostic evaluation may be warranted.
                        </div>

                        **Recommended Actions:**
                        - Schedule comprehensive cardiac evaluation  
                        - Consider additional diagnostic tests  
                        - Review and optimize cardiovascular risk factors  
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="info-box">
                        The predictive model suggests a <strong>lower probability</strong> of cardiovascular disease. 
                        Preventive care and regular monitoring are still important.
                        </div>

                        **Recommendations:**
                        - Continue regular health monitoring  
                        - Maintain heart-healthy lifestyle habits  
                        - Stay proactive with preventive measures  
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<p class="small-text">⚠️ <strong>Important:</strong> Educational model only. Not for real clinical decisions.</p>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="card-title">📋 Patient Data Summary</p>', unsafe_allow_html=True)

                df_inputs = pd.DataFrame(
                    {
                        "Parameter": [
                            "Age",
                            "Sex",
                            "Resting Blood Pressure",
                            "Serum Cholesterol",
                            "Fasting Blood Sugar > 120",
                            "Maximum Heart Rate",
                            "ST Depression (Oldpeak)",
                            "Chest Pain Type",
                            "Resting ECG",
                            "Exercise-Induced Angina",
                            "ST Slope",
                        ],
                        "Value": [
                            f"{age} years",
                            sex,
                            f"{resting_bp} mm Hg",
                            f"{cholesterol} mg/dl",
                            "Yes" if fasting_bs == 1 else "No",
                            f"{max_hr} bpm",
                            f"{oldpeak}",
                            chest_pain_type,
                            resting_ecg,
                            "Yes" if exercise_angina == "Y" else "No",
                            st_slope,
                        ],
                    }
                )
                st.dataframe(df_inputs, hide_index=True, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.error(f"❌ API Error: {resp.status_code} - {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the prediction API. Ensure the backend is running.")
        except Exception as e:
            st.error(f"❌ Request failed: {e}")

# ---------------- EXPERIMENTS TABLE ----------------
st.markdown("---")
st.header("Model Experiments (16 runs)")

metrics_path = "../metrics/all_experiments.csv"

try:
    metrics_df = pd.read_csv(metrics_path)
except Exception as e:
    st.error(f"Could not load metrics from {metrics_path}: {e}")
else:
    if "f1_score" not in metrics_df.columns:
        st.error("Column 'f1_score' not found in metrics CSV.")
    else:
        best_idx = metrics_df["f1_score"].idxmax()
        best_row = metrics_df.loc[best_idx]

        st.subheader("Best experiment")
        st.write(f"Model: **{best_row['model']}**")
        st.write(f"Use PCA: **{best_row['pca_applied']}**")
        st.write(f"Tuning: **{best_row['hyperparameter_tuning']}**")
        st.write(f"F1-score: **{best_row['f1_score']:.4f}**")

        def highlight_best(row):
            if row.name == best_idx:
                return ["background-color: #d1ffd6"] * len(row)
            else:
                return [""] * len(row)

        st.subheader("All 16 experiments")
        st.dataframe(metrics_df.style.apply(highlight_best, axis=1))
