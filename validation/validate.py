import pandas as pd


def validate_data(df):

    print("\n========== DATA VALIDATION ==========")

    # Shape
    print("Rows :", df.shape[0])
    print("Columns :", df.shape[1])

    # Missing Values
    print("\nMissing Values")
    print(df.isnull().sum())

    # Duplicate Records
    duplicates = df.duplicated().sum()
    print("\nDuplicate Records :", duplicates)

    # Data Types
    print("\nData Types")
    print(df.dtypes)

    print("\nValidation Completed")

    return df