import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    if 'rating' in df.columns:
        df = df[df['rating'].between(1, 5)]
    return df

if __name__ == "__main__":
    data = load_and_clean("data/sample_ratings.csv")
    print("Rows:", len(data))
    print(data.head())
