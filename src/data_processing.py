import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.cluster import KMeans


def create_aggregate_features(df):
    """
    Create customer-level aggregate transaction features.
    """

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std")
        )
        .reset_index()
    )

    return agg_df


def extract_datetime_features(df):
    """
    Extract useful datetime features.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["transaction_day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["transaction_month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["transaction_year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


def merge_aggregate_features(df):
    """
    Merge aggregate customer features back to transaction data.
    """

    agg_df = create_aggregate_features(df)

    df = df.merge(
        agg_df,
        on="CustomerId",
        how="left"
    )

    return df


df = pd.read_csv("data/raw/data.csv")

df = extract_datetime_features(df)

df = merge_aggregate_features(df)

required_columns = [
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
]

for col in required_columns:
    print(col, col in df.columns)