import pandas as pd


def stochastic(df: pd.DataFrame, k_period: int = 7, d_period: int = 10) -> pd.DataFrame:
    low = df["Low"].rolling(k_period).min()
    high = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low) / (high - low).replace(0, pd.NA)
    return pd.DataFrame({"K": k, "D": k.rolling(d_period).mean()})

