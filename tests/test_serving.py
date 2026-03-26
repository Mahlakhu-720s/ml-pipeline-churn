"""
tests/test_serving.py
----------------------
Unit tests for the FastAPI serving layer.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# ---------------------------------------------------------------------------
# Tests: feature vector construction
# ---------------------------------------------------------------------------

def test_feature_vector_month_to_month_contract():
    from src.serving.api import build_feature_vector, PredictionRequest
    req = PredictionRequest(
        tenure=6,
        monthly_charges=80.0,
        total_charges=480.0,
        contract="Month-to-month",
        internet_service="Fiber optic",
        payment_method="Electronic check",
    )
    fv = build_feature_vector(req)
    # contract_months index=5 → should be 1 for Month-to-month
    assert fv[0, 5] == 1


def test_feature_vector_two_year_contract():
    from src.serving.api import build_feature_vector, PredictionRequest
    req = PredictionRequest(
        tenure=24,
        monthly_charges=50.0,
        total_charges=1200.0,
        contract="Two year",
        internet_service="DSL",
        payment_method="Bank transfer (automatic)",
    )
    fv = build_feature_vector(req)
    assert fv[0, 5] == 24


def test_charge_per_tenure_in_feature_vector():
    from src.serving.api import build_feature_vector, PredictionRequest
    req = PredictionRequest(
        tenure=9,
        monthly_charges=90.0,
        total_charges=810.0,
        contract="Month-to-month",
        internet_service="DSL",
        payment_method="Mailed check",
    )
    fv = build_feature_vector(req)
    expected = 90.0 / (9 + 1)
    assert abs(fv[0, 4] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Tests: risk tier logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prob,expected", [
    (0.0, "LOW"),
    (0.15, "LOW"),
    (0.299, "LOW"),
    (0.30, "MEDIUM"),
    (0.45, "MEDIUM"),
    (0.599, "MEDIUM"),
    (0.60, "HIGH"),
    (0.85, "HIGH"),
    (1.0, "HIGH"),
])
def test_risk_tier_parametrized(prob, expected):
    from src.serving.api import get_risk_tier
    assert get_risk_tier(prob) == expected


# ---------------------------------------------------------------------------
# Tests: validation (Pydantic)
# ---------------------------------------------------------------------------

def test_invalid_tenure_raises():
    from src.serving.api import PredictionRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PredictionRequest(
            tenure=-1,  # below minimum
            monthly_charges=60.0,
            total_charges=720.0,
            contract="Month-to-month",
            internet_service="DSL",
            payment_method="Electronic check",
        )


def test_invalid_tenure_too_high():
    from src.serving.api import PredictionRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PredictionRequest(
            tenure=100,  # above maximum of 72
            monthly_charges=60.0,
            total_charges=6000.0,
            contract="One year",
            internet_service="DSL",
            payment_method="Mailed check",
        )


# ---------------------------------------------------------------------------
# Tests: drift detection
# ---------------------------------------------------------------------------

def test_no_drift_on_same_data():
    from src.monitoring.drift import compute_psi, compute_ks_test
    np.random.seed(0)
    data = np.random.normal(50, 10, 500)
    psi = compute_psi(data, data)
    ks_stat, ks_p = compute_ks_test(data, data)
    assert psi < 0.01
    assert ks_p > 0.05  # should NOT reject null hypothesis


def test_drift_detected_on_shifted_distribution():
    from src.monitoring.drift import compute_psi
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(5, 1, 1000)  # large mean shift
    psi = compute_psi(ref, cur)
    assert psi > 0.2, f"Expected PSI > 0.2, got {psi}"


def test_full_drift_report_structure(reference_production_data):
    from src.monitoring.drift import run_drift_detection
    import tempfile, os
    ref_df, cur_df = reference_production_data
    numeric_features = ["tenure", "MonthlyCharges"]

    # Only run if these columns exist
    available = [f for f in numeric_features if f in ref_df.columns]
    if not available:
        pytest.skip("Numeric features not in synthetic dataset columns")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        report = run_drift_detection(ref_df, cur_df, available, output_path=tmp_path)
        assert "features" in report
        assert "summary" in report
        assert "features_checked" in report["summary"]
        assert "trigger_retrain" in report["summary"]
    finally:
        os.unlink(tmp_path)
