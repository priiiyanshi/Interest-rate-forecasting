import pandas as pd

def clean_rate_data(df):
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.sort_index()
    df["rate"] = df["rate"].astype(float)
    return df
