# ======================= IMPORTS =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ======================= PAGE CONFIG =======================
st.set_page_config(
    page_title="CVD Risk Prediction System",
    page_icon="❤️",
    layout="wide"
)

# ======================= LOAD DATA =======================
@st.cache_data
def load_data():
    df = pd.read_csv("CVD_dataset.csv")
    return df

df = load_data()

# ======================= DATA CLEANING =======================
df = df.rename(columns={
    "Smoki0g Status": "Smoking Status",
    "Famil1 Histor1 of CVD": "Family History of CVD",
    "Blood Pressure (mmHg)": "Blood Pressure",
})

# Split BP column if needed
if df["Blood Pressure"].dtype == "object":
    df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
    df['BP_Systolic'] = pd.to_numeric(df['BP_Systolic'], errors='coerce')
    df['BP_Diastolic'] = pd.to_numeric(df['BP_Diastolic'], errors='coerce')

df = df.drop(columns=['Blood Pressure', 'Height (cm)'], errors='ignore')

# Fill missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# ======================= ENCODING =======================
le = LabelEncoder()
df['CVD Risk Level Encoded'] = le.fit_transform(df['CVD Risk Level'])

df = df.drop(columns=['BP_Systolic', 'BP_Diastolic'], errors='ignore')

df_encoded = pd.get_dummies(
    df,
    columns=['Sex', 'Physical Activity Level', 'Blood Pressure Category'],
    drop_first=True
)

X = df_encoded.drop(columns=['CVD Risk Level', 'CVD Risk Level Encoded', 'CVD Risk Score'])
y = df_encoded['CVD Risk Level Encoded']

# ======================= TRAIN TEST SPLIT =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================= SCALING =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================= MODELS =======================
lr = LogisticRegression(max_iter=300, class_weight='balanced')
dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False
)

# ======================= TRAIN MODELS =======================
lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ======================= SIDEBAR =======================
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio(
    "",
    ["Predict CVD Risk", "Graphs", "Dataset View", "Model Comparison", "Project Info"],
    label_visibility="collapsed"
)

# ======================= PREDICTION =======================
if section == "Predict CVD Risk":

    st.title("❤️ CVD Risk Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 45)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        chol = st.slider("Total Cholesterol", 100, 350, 180)
        ldl = st.slider("Estimated LDL", 50, 250, 120)

    with col2:
        sugar = st.slider("Fasting Blood Sugar", 60, 250, 100)
        sex = st.selectbox("Sex", ["Male", "Female"])
        activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        bp_cat = st.selectbox("Blood Pressure Category", ["Normal", "Elevated", "Hypertension"])

    model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]
    )

    input_data = {
        "Age": age,
        "BMI": bmi,
        "Total Cholesterol (mg/dL)": chol,
        "Fasting Blood Sugar (mg/dL)": sugar,
        "Estimated LDL (mg/dL)": ldl,
        "Sex_Male": 1 if sex == "Male" else 0,
        "Physical Activity Level_Low": 1 if activity == "Low" else 0,
        "Physical Activity Level_Moderate": 1 if activity == "Moderate" else 0,
        "Blood Pressure Category_Elevated": 1 if bp_cat == "Elevated" else 0,
        "Blood Pressure Category_Hypertension": 1 if bp_cat == "Hypertension" else 0
    }

    input_df = pd.DataFrame([input_data]).reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    if st.button("🔍 Predict"):
        if model_name == "Random Forest":
            pred = rf.predict(input_df)
        elif model_name == "Logistic Regression":
            pred = lr.predict(input_scaled)
        elif model_name == "Decision Tree":
            pred = dt.predict(input_df)
        else:
            pred = xgb.predict(input_df)

        result = le.inverse_transform(pred)[0]
        st.success(f"Predicted CVD Risk Level: {result}")

# ======================= GRAPHS =======================
elif section == "Graphs":

    st.title("📈 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df['BMI'], kde=True, ax=ax)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ======================= DATASET VIEW =======================
elif section == "Dataset View":

    st.title("📂 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

# ======================= MODEL COMPARISON =======================
elif section == "Model Comparison":

    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
        "Accuracy": [
            accuracy_score(y_test, lr.predict(X_test_scaled)),
            accuracy_score(y_test, dt.predict(X_test)),
            accuracy_score(y_test, rf.predict(X_test)),
            accuracy_score(y_test, xgb.predict(X_test))
        ]
    })

    st.title("📊 Model Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=comparison_df, ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ======================= PROJECT INFO =======================
else:

    st.title("📘 Project Information")
    st.markdown("""
    **Project:** Cardiovascular Disease Risk Prediction  
    **Tech Stack:** Python, ML, XGBoost, Streamlit  
    **Outcome:** Real-time healthcare risk prediction system  
    """)
