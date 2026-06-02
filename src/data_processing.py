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
from sklearn.preprocessing import StandardScaler
from category_encoders.woe import WOEEncoder


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


def create_rfm_features(df):
    """
    Create Recency, Frequency and Monetary metrics.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum")
        )
        .reset_index()
    )

    return rfm

rfm = create_rfm_features(df)

print(rfm.head())

def cluster_customers_rfm(rfm_df):
    """
    Segment customers using KMeans.
    """

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm_df[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm_df["cluster"] = (
        kmeans.fit_predict(rfm_scaled)
    )

    return rfm_df

rfm = cluster_customers_rfm(rfm)

print(
    rfm["cluster"].value_counts()
)

def assign_high_risk_label(rfm_df):
    """
    Identify least-engaged cluster and
    assign is_high_risk.
    """

    cluster_summary = (
        rfm_df.groupby("cluster")
        .agg(
            {
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean"
            }
        )
    )

    print(cluster_summary)

    cluster_summary["risk_score"] = (
    cluster_summary["Recency"]
    - cluster_summary["Frequency"]
    - cluster_summary["Monetary"]
    )

    high_risk_cluster = (
        cluster_summary["risk_score"]
        .idxmax()
    )

    rfm_df["is_high_risk"] = (
    rfm_df["cluster"]
    == high_risk_cluster
    ).astype(int)

    return rfm_df

rfm = assign_high_risk_label(rfm)

def merge_target_variable(
    transaction_df,
    rfm_df
):
    """
    Merge target variable back to
    transaction dataset.
    """

    return transaction_df.merge(
        rfm_df[
            [
                "CustomerId",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )

processed_df = merge_target_variable(
    df,
    rfm
)

print(
    processed_df["is_high_risk"]
    .value_counts()
)


numeric_features = [
    "Amount",
    "Value",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount"
]

categorical_features = [
    "ProductCategory",
    "ChannelId",
    "PricingStrategy",
    "ProviderId"
]

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)

feature_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        )
    ]
)


def apply_woe_encoding(
    X_train,
    y_train,
    categorical_cols
):
    """
    Apply Weight of Evidence encoding.
    """

    woe_encoder = WOEEncoder(
        cols=categorical_cols
    )

    X_train_woe = woe_encoder.fit_transform(
        X_train,
        y_train
    )

    return X_train_woe, woe_encoder

def calculate_information_value(
    df,
    target,
    feature
):
    """
    Calculate Information Value.
    """
    pass

X_processed = feature_pipeline.fit_transform(
    processed_df
)

processed_df.to_csv(
    "data/processed/processed_data.csv",
    index=False
)