"""
tests/conftest.py
------------------
Shared pytest fixtures for the test suite.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session")
def synthetic_dataset():
    """
    Session-scoped synthetic binary classification dataset.
    Reused across all tests to avoid regenerating data.
    """
    X, y = make_classification(
        n_samples=300,
        n_features=11,
        n_informative=7,
        n_redundant=2,
        weights=[0.735, 0.265],
        random_state=42,
    )
    feature_names = [
        "tenure", "MonthlyCharges", "TotalCharges", "num_products",
        "charge_per_tenure", "contract_months", "SeniorCitizen",
        "InternetService_idx", "PaymentMethod_idx", "Partner_idx", "Dependents_idx",
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["Churn"] = y
    df["gender"] = np.random.choice(["Male", "Female"], len(y))
    df["SeniorCitizen"] = np.random.choice([0, 1], len(y), p=[0.85, 0.15])
    df["Partner"] = np.random.choice(["Yes", "No"], len(y))
    return df, X, y


@pytest.fixture(scope="session")
def trained_model(synthetic_dataset):
    """A small fitted RandomForest model for serving tests."""
    _, X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def reference_production_data(synthetic_dataset):
    """Split synthetic data into reference and current for drift tests."""
    df, X, y = synthetic_dataset
    n = len(df)
    ref = df.iloc[:n // 2].copy()
    cur = df.iloc[n // 2:].copy()
    return ref, cur
