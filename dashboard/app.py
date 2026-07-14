import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Healthcare Data Engineering Pipeline",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

[data-testid="stMetricValue"]{
    font-size:32px;
    font-weight:bold;
    color:#0E76A8;
}

h1,h2,h3{
    color:#0E1117;
}

section[data-testid="stSidebar"]{
    background:#1F2937;
}

section[data-testid="stSidebar"] *{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ---------------- #

conn = sqlite3.connect("database/cvd.db")

# ---------------- LOAD MODELS ---------------- #

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/features.pkl")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("❤️ CVD Pipeline")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 SQL Analytics",
        "🤖 Prediction",
        "📂 Dataset",
        "📈 Visualizations",
        "ℹ About"
    ]
)
if page=="🏠 Dashboard":

    st.title("Healthcare Data Engineering Pipeline")

    st.caption("ETL • SQL • Machine Learning • Streamlit")

    total = pd.read_sql(
        "SELECT COUNT(*) Total FROM patients",
        conn
    )

    avg_age = pd.read_sql(
        "SELECT ROUND(AVG(Age),2) Age FROM patients",
        conn
    )

    avg_bmi = pd.read_sql(
        "SELECT ROUND(AVG(BMI),2) BMI FROM patients",
        conn
    )

    high = pd.read_sql(
        '''
        SELECT COUNT(*) Total
        FROM patients
        WHERE "CVD Risk Level"='HIGH'
        ''',
        conn
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Patients",int(total.iloc[0,0]))
    c2.metric("Average Age",float(avg_age.iloc[0,0]))
    c3.metric("Average BMI",float(avg_bmi.iloc[0,0]))
    c4.metric("High Risk",int(high.iloc[0,0]))

    st.divider()

    risk = pd.read_sql("""
    SELECT
    "CVD Risk Level",
    COUNT(*) Total
    FROM patients
    GROUP BY "CVD Risk Level"
    """,conn)

    fig = px.pie(
        risk,
        names="CVD Risk Level",
        values="Total",
        title="Risk Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)