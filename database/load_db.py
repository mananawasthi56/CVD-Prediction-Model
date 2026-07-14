import sqlite3
import pandas as pd


def load_to_database():

    # Read cleaned dataset
    df = pd.read_csv("data/processed/cleaned_cvd.csv")

    # Create Database
    conn = sqlite3.connect("database/cvd.db")

    # Store table
    df.to_sql(
        "patients",
        conn,
        if_exists="replace",
        index=False
    )

    conn.commit()
    conn.close()

    print("Database Created Successfully")
    print("Table Name : patients")
    print("Database Location : database/cvd.db")