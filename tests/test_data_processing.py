import pandas as pd
from src.data_processing import (
    create_aggregate_features,
    extract_datetime_features,
    create_rfm_features
)

def test_create_aggregate_features():

    data = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, 200, 50],
        "TransactionId": ["t1", "t2", "t3"]
    })

    result = create_aggregate_features(data)

    # check output columns exist
    assert "total_transaction_amount" in result.columns
    assert "avg_transaction_amount" in result.columns
    assert "transaction_count" in result.columns

    # check correctness
    row = result[result["CustomerId"] == 1].iloc[0]

    assert row["total_transaction_amount"] == 300
    assert row["transaction_count"] == 2


def test_extract_datetime_features():

    data = pd.DataFrame({
        "TransactionStartTime": [
            "2024-01-01 10:30:00",
            "2024-01-02 15:45:00"
        ]
    })

    result = extract_datetime_features(data)

    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_month" in result.columns
    assert "transaction_year" in result.columns

    assert result["transaction_hour"].iloc[0] == 10
    assert result["transaction_day"].iloc[1] == 2

