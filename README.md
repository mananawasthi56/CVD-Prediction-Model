# ❤️ Healthcare Data Engineering Pipeline with Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![SQLite](https://img.shields.io/badge/Database-SQLite-green)
![SQL](https://img.shields.io/badge/SQL-Analytics-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-yellow)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-success)

---

## 📌 Project Overview

This project demonstrates an **End-to-End Healthcare Data Engineering Pipeline** for Cardiovascular Disease (CVD) Risk Prediction. It performs data ingestion, cleaning, validation, feature engineering, SQL analytics, machine learning model training, and interactive visualization through a Streamlit dashboard.

The project follows a production-style ETL workflow where raw healthcare data is transformed into structured datasets, stored in a SQLite database, analyzed using SQL, and finally used for machine learning predictions.

---

## 🚀 Project Architecture

```
Raw Dataset
      │
      ▼
Data Ingestion
      │
      ▼
Data Cleaning
      │
      ▼
Data Validation
      │
      ▼
Feature Engineering
      │
      ▼
SQLite Database
      │
      ▼
SQL Analytics
      │
      ▼
Machine Learning
      │
      ▼
Streamlit Dashboard
```

---

# ✨ Features

- Automated ETL Pipeline
- Data Cleaning & Validation
- Feature Engineering
- SQLite Database Integration
- SQL Analytics
- Machine Learning Pipeline
- Multiple ML Models
- Model Serialization using Joblib
- Interactive Streamlit Dashboard
- Data Visualization

---

# 🛠 Tech Stack

### Programming

- Python

### Data Engineering

- Pandas
- NumPy
- SQLite
- SQL

### Machine Learning

- Scikit-Learn
- XGBoost
- Joblib

### Visualization

- Plotly
- Matplotlib
- Streamlit

---

# 📂 Project Structure

```
CVD-Prediction-Model
│
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── database/
├── ingestion/
├── logs/
├── models/
├── pipeline/
├── preprocessing/
├── sql/
├── validation/
│
├── Code.py
├── README.md
└── requirements.txt
```

---

# ⚙ ETL Pipeline

### Data Ingestion

- Reads raw healthcare dataset

### Data Cleaning

- Missing value handling
- Column standardization
- Blood Pressure parsing

### Data Validation

- Duplicate check
- Missing value validation
- Data type validation

### Feature Engineering

- Label Encoding
- One-Hot Encoding
- Train-Test Split
- Feature Scaling

### Database

- SQLite Database
- SQL Queries
- Analytical Reports

---

# 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Best Model:
**XGBoost**

---

# 📊 SQL Analytics

Implemented SQL queries for:

- Total Patients
- Average Age
- Average BMI
- Risk Distribution
- Smoking Statistics
- Cholesterol Analysis

---

# 📈 Dashboard

The Streamlit dashboard provides:

- Healthcare KPIs
- Risk Distribution
- SQL Analytics
- Dataset Explorer
- ML Predictions
- Interactive Charts

---

# ▶ Installation

```bash
git clone https://github.com/mananawasthi56/CVD-Prediction-Model.git

cd CVD-Prediction-Model
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run ETL Pipeline

```bash
py -3.13 -m pipeline.run_pipeline
```

Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 🎯 Future Improvements

- Docker Deployment
- Apache Airflow
- PostgreSQL Support
- AWS Deployment
- MLflow Integration
- CI/CD using GitHub Actions

---

# 👨‍💻 Developed By

**Manan Awasthi**

Lovely Professional University

Data Science Undergraduate

---

# ⭐ If you found this project useful, don't forget to star the repository.