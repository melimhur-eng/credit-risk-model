import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def load_data(path):
    return pd.read_csv(path)


def prepare_data(df):

    X = df.drop(
        columns=["is_high_risk"]
    )

    y = df["is_high_risk"]

    return X, y

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_logistic_regression(
    X_train,
    y_train
):

    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }

    grid_search = GridSearchCV(
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42
        ),
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search

def train_random_forest(
    X_train,
    y_train
):

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(
            random_state=42
        ),
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba)
    }

mlflow.set_experiment("credit-risk-model")

def log_models(X_train, X_test, y_train, y_test):

    mlflow.set_experiment("credit-risk-model")

    models = {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba)
        }

        with mlflow.start_run(run_name=name):

            mlflow.log_params(model.best_params_)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                model.best_estimator_,
                "model"
            )

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model.best_estimator_

    # Register best model
    mlflow.sklearn.log_model(
        best_model,
        "best_model",
        registered_model_name="credit_risk_model"
    )


def main():
    df = load_data("data/processed/processed_data.csv")

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    log_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()