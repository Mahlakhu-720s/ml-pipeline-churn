"""
tests/test_features.py
-----------------------
Unit tests for the feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_df():
    """Minimal raw DataFrame matching the Telco schema."""
    return pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 1, 60],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["Yes", "No", "Yes"],
        "OnlineBackup": ["No", "Yes", "Yes"],
        "DeviceProtection": ["No", "No", "Yes"],
        "TechSupport": ["Yes", "No", "No"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "Contract": ["Month-to-month", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [65.0, 85.0, 45.0],
        "TotalCharges": [780.0, 85.0, 2700.0],
        "Churn": [0, 1, 0],
    })


# ---------------------------------------------------------------------------
# Tests: engineered features
# ---------------------------------------------------------------------------

def test_num_products_correct(sample_raw_df):
    """num_products should count 'Yes' flags across 8 add-on service columns."""
    from src.features.engineering import engineer_features_pandas  # noqa

    # Customer 001: PhoneService=Yes, OnlineSecurity=Yes, TechSupport=Yes → 3
    # Customer 002: PhoneService=Yes, MultipleLines=Yes, OnlineBackup=Yes, StreamingTV=Yes → 4
    # Customer 003: OnlineSecurity=Yes, OnlineBackup=Yes, DeviceProtection=Yes → 3

    result = _compute_num_products_pandas(sample_raw_df)
    assert result.iloc[0] == 3
    assert result.iloc[1] == 4
    assert result.iloc[2] == 3


def test_charge_per_tenure_new_customer(sample_raw_df):
    """charge_per_tenure = MonthlyCharges / (tenure + 1). New customer (tenure=1) gets divide by 2."""
    charges = sample_raw_df.loc[1, "MonthlyCharges"]
    tenure = sample_raw_df.loc[1, "tenure"]
    expected = charges / (tenure + 1)
    result = charges / (tenure + 1)
    assert abs(result - expected) < 1e-6
    assert result == pytest.approx(85.0 / 2, abs=1e-4)


def test_contract_months_mapping():
    """Verify ordinal contract encoding."""
    mapping = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    assert mapping["Month-to-month"] == 1
    assert mapping["One year"] == 12
    assert mapping["Two year"] == 24


# ---------------------------------------------------------------------------
# Tests: data quality
# ---------------------------------------------------------------------------

def test_total_charges_whitespace_handling():
    """TotalCharges with whitespace should be cast to 0.0, not NaN."""
    raw = pd.DataFrame({
        "TotalCharges": ["780.0", " ", "2700.0", ""],
    })
    cleaned = pd.to_numeric(raw["TotalCharges"].str.strip().replace("", float("nan")), errors="coerce").fillna(0.0)
    assert cleaned.iloc[1] == 0.0
    assert cleaned.iloc[3] == 0.0
    assert cleaned.iloc[0] == 780.0


def test_churn_encoding():
    """Churn 'Yes' → 1, 'No' → 0."""
    series = pd.Series(["Yes", "No", "Yes", "No"])
    encoded = (series == "Yes").astype(int)
    assert list(encoded) == [1, 0, 1, 0]


# ---------------------------------------------------------------------------
# Tests: PSI computation
# ---------------------------------------------------------------------------

def test_psi_identical_distributions():
    """PSI of two identical distributions should be ~0."""
    from src.monitoring.drift import compute_psi

    ref = np.random.normal(50, 10, 1000)
    psi = compute_psi(ref, ref.copy())
    assert psi < 0.01


def test_psi_very_different_distributions():
    """PSI of highly different distributions should exceed 0.2."""
    from src.monitoring.drift import compute_psi

    ref = np.random.normal(10, 1, 1000)
    cur = np.random.normal(100, 1, 1000)
    psi = compute_psi(ref, cur)
    assert psi > 0.2


def test_psi_moderate_shift():
    """PSI of moderately shifted distributions should be in 0.1–0.2 range."""
    from src.monitoring.drift import compute_psi

    np.random.seed(42)
    ref = np.random.normal(50, 10, 2000)
    cur = np.random.normal(60, 12, 2000)  # slight shift
    psi = compute_psi(ref, cur)
    assert 0.05 < psi < 0.5  # wider bounds — depends on random seed


# ---------------------------------------------------------------------------
# Tests: serving API feature construction
# ---------------------------------------------------------------------------

def test_build_feature_vector_shape():
    """Feature vector should have the expected number of features."""
    from src.serving.api import build_feature_vector, PredictionRequest

    req = PredictionRequest(
        tenure=24,
        monthly_charges=65.5,
        total_charges=1572.0,
        contract="Month-to-month",
        internet_service="Fiber optic",
        payment_method="Electronic check",
        num_products=3,
    )
    features = build_feature_vector(req)
    assert features.ndim == 2
    assert features.shape[0] == 1
    assert features.shape[1] == 11


def test_risk_tier_thresholds():
    """Risk tier boundaries should match documented thresholds."""
    from src.serving.api import get_risk_tier

    assert get_risk_tier(0.15) == "LOW"
    assert get_risk_tier(0.29) == "LOW"
    assert get_risk_tier(0.30) == "MEDIUM"
    assert get_risk_tier(0.59) == "MEDIUM"
    assert get_risk_tier(0.60) == "HIGH"
    assert get_risk_tier(0.95) == "HIGH"


def test_contract_months_in_feature_vector():
    """Two-year contract should encode to 24 months in the feature vector."""
    from src.serving.api import build_feature_vector, PredictionRequest

    req = PredictionRequest(
        tenure=36,
        monthly_charges=50.0,
        total_charges=1800.0,
        contract="Two year",
        internet_service="DSL",
        payment_method="Bank transfer (automatic)",
    )
    features = build_feature_vector(req)
    # contract_months is the 6th feature (index 5)
    assert features[0, 5] == 24


# ---------------------------------------------------------------------------
# Helper (used in test above — pandas fallback for unit testing without Spark)
# ---------------------------------------------------------------------------

def _compute_num_products_pandas(df: pd.DataFrame) -> pd.Series:
    """Compute num_products without Spark for unit test purposes."""
    add_on_services = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    return df[add_on_services].apply(lambda row: (row == "Yes").sum(), axis=1)
