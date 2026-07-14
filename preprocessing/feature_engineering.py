import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def feature_engineering():

    # Read cleaned dataset
    df = pd.read_csv("data/processed/cleaned_cvd.csv")

    # Label Encoding
    le = LabelEncoder()
    df["CVD Risk Level Encoded"] = le.fit_transform(df["CVD Risk Level"])

    # Drop unwanted columns
    df.drop(
        columns=[
            "CVD Risk Level",
            "CVD Risk Score"
        ],
        inplace=True,
        errors="ignore"
    )

    # One Hot Encoding
    df = pd.get_dummies(
        df,
        columns=[
            "Sex",
            "Physical Activity Level",
            "Blood Pressure Category"
        ],
        drop_first=True
    )

    # Features
    X = df.drop(columns=["CVD Risk Level Encoded"])

    # Target
    y = df["CVD Risk Level Encoded"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save Scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Save Label Encoder
    joblib.dump(le, "models/label_encoder.pkl")

    # Save Feature Names
    joblib.dump(list(X.columns), "models/features.pkl")

    print("Feature Engineering Completed")

    return X_train, X_test, y_train, y_test