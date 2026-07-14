from ingestion.ingest import load_data
from preprocessing.cleandata import clean_data
from validation.validate import validate_data
from database.load_db import load_to_database
from database.query_db import execute_queries
from preprocessing.feature_engineering import feature_engineering
from models.train_model import train_models


def main():

    # Step 1
    df = load_data()

    # Step 2
    df = clean_data(df)

    # Step 3
    validate_data(df)

    print("\nPipeline Executed Successfully")


if __name__ == "__main__":
    main()

load_to_database()
execute_queries()
X_train, X_test, y_train, y_test = feature_engineering()
train_models()