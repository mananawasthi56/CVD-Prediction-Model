import sqlite3
import pandas as pd


def execute_queries():

    conn = sqlite3.connect("database/cvd.db")

    queries = {

        "Total Patients":
        "SELECT COUNT(*) AS Total FROM patients",

        "Average Age":
        'SELECT ROUND(AVG(Age),2) AS Average_Age FROM patients',

        "Average BMI":
        'SELECT ROUND(AVG(BMI),2) AS Average_BMI FROM patients',

        "Risk Distribution":
        'SELECT "CVD Risk Level", COUNT(*) AS Total FROM patients GROUP BY "CVD Risk Level"'

    }

    for title, query in queries.items():

        print("\n" + "="*50)
        print(title)
        print("="*50)

        df = pd.read_sql(query, conn)

        print(df)

    conn.close()