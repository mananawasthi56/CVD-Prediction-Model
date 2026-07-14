import pandas as pd


def clean_data(df):
    """
    Clean the raw CVD dataset
    """

    # Rename incorrect column names
    df = df.rename(columns={
        "Smoki0g Status": "Smoking Status",
        "Famil1 Histor1 of CVD": "Family History of CVD",
        "Blood Pressure (mmHg)": "Blood Pressure"
    })

    # Split Blood Pressure
    if "Blood Pressure" in df.columns and df["Blood Pressure"].dtype == "object":
        bp = df["Blood Pressure"].str.split("/", expand=True)

        df["BP_Systolic"] = pd.to_numeric(bp[0], errors="coerce")
        df["BP_Diastolic"] = pd.to_numeric(bp[1], errors="coerce")

    # Drop unwanted columns
    df.drop(columns=["Blood Pressure", "Height (cm)"], errors="ignore", inplace=True)

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Data Cleaning Completed")

    # Save cleaned dataset
    df.to_csv("data/processed/cleaned_cvd.csv", index=False)

    print("Processed dataset saved successfully.")

    return df