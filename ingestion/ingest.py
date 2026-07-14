import pandas as pd


def load_data():
    """
    Load raw dataset
    """

    file_path = "data/raw/CVD_dataset.csv"

    df = pd.read_csv(file_path)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


if __name__ == "__main__":
    load_data()