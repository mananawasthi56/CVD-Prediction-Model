# CVD-Prediction-Model
# ❤️ CVD Risk Predictor: Cardiovascular Disease Risk Prediction Using Machine Learning

<grok-card data-id="0f433d" data-type="image_card"></grok-card>


**Project for Predictive Analytics (INT234)**  


**Submitted by:**  
MANAN AWASTHI 

Course Code: INT234  


**Discipline of CSE/IT**  
**Lovely School of Computer Science and Engineering**  
**Lovely Professional University, Phagwara**  

## Project Overview

This project develops a machine learning system to predict **Cardiovascular Disease (CVD) risk levels** (Low, Medium, High) based on patient health parameters. It uses a large synthetic dataset from Kaggle and trains multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost). The system deploys as an interactive **Streamlit web application** for real-time predictions, visualizations, and model comparison.

**Tech Stack:**  
Python • Pandas • Scikit-learn • XGBoost • Matplotlib/Seaborn • Streamlit

<grok-card data-id="6c0183" data-type="image_card"></grok-card>



<grok-card data-id="fc1c11" data-type="image_card"></grok-card>


Key Features:
- **Real-time Prediction** — Users input age, BMI, cholesterol, blood sugar, etc., and get instant risk level.
- **Model Selection** — Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Visualizations** — Histograms, correlation heatmap, dataset overview.
- **Model Comparison** — Accuracy and metrics table.

CVD is the leading cause of death globally (WHO). This tool aids early risk assessment and preventive healthcare.

<grok-card data-id="d3a8af" data-type="image_card"></grok-card>



<grok-card data-id="5c7554" data-type="image_card"></grok-card>


## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models & Results](#models--results)
- [Streamlit App Screenshots](#streamlit-app-screenshots)
- [Installation & Run](#installation--run)
- [Usage](#usage)
- [Future Scope](#future-scope)
- [References](#references)

## Dataset

- **Name**: Cardiovascular Diseases Risk Prediction Dataset
- **Source**: [Kaggle - alphiree](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset)
- **Size**: ~308,854 rows, multiple features (Age, BMI, Cholesterol, Blood Sugar, etc.)
- **Target**: CVD Risk Level (Low, Medium, High) - Multi-class classification

## Preprocessing

- Cleaned column names and parsed blood pressure.
- Imputed missing values (median/mode).
- Outlier treatment via IQR winsorization.
- One-hot encoding for categorical features.
- Label encoding for target.

## Models & Results

Trained models:
- Logistic Regression
- Decision Tree
- Random Forest (with/without SMOTE)
- **XGBoost** (Best model: 67.65% accuracy)

<grok-card data-id="3aa141" data-type="image_card"></grok-card>



<grok-card data-id="80c532" data-type="image_card"></grok-card>


Challenge: Minority High-risk class detection due to imbalance.

## Streamlit App Screenshots

<grok-card data-id="6a23a9" data-type="image_card"></grok-card>



<grok-card data-id="da3741" data-type="image_card"></grok-card>



<grok-card data-id="c6f7ea" data-type="image_card"></grok-card>



<grok-card data-id="fdd9c4" data-type="image_card"></grok-card>



<grok-card data-id="81ab94" data-type="image_card"></grok-card>


## Installation & Run

```bash
git clone https://github.com/[mananawasthi56]/CVD-Risk-Prediction.git
cd CVD-Risk-Prediction
pip install -r requirements.txt
streamlit run app.py
