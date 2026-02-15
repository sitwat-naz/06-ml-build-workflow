import pandas as pd

def preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    return df
