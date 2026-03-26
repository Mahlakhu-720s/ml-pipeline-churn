"""
src/data/validate.py
---------------------
Data quality validation for the churn pipeline.

Runs a suite of expectations against the raw ingested data before it
enters the feature engineering stage. Any critical failure halts the
pipeline and raises a DataQualityError.

Validation rules are inspired by Great Expectations conventions but
implemented without the GE dependency for portability.

Severity levels
---------------
CRITICAL  — pipeline halts immediately (e.g. missing required columns)
WARNING   — logged but pipeline continues (e.g. unexpected null rate)
INFO      — informational only
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    rule_name: str
    passed: bool
    severity: Severity
    detail: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ValidationReport:
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == Severity.CRITICAL)

    @property
    def n_critical_failures(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == Severity.CRITICAL)

    @property
    def n_warnings(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == Severity.WARNING)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "n_critical_failures": self.n_critical_failures,
            "n_warnings": self.n_warnings,
            "results": [
                {
                    "rule": r.rule_name,
                    "passed": r.passed,
                    "severity": r.severity.value,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }


class DataQualityError(RuntimeError):
    """Raised when critical data quality checks fail."""


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "InternetService", "Contract", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

NUMERIC_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (0, 200),
    "TotalCharges": (0, 15000),
    "SeniorCitizen": (0, 1),
}

CATEGORICAL_ALLOWED = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
}

NULL_RATE_THRESHOLD = 0.05   # 5% null rate triggers WARNING
DUPLICATE_ID_THRESHOLD = 0   # Any duplicate customerID is CRITICAL


def validate(df: pd.DataFrame, output_path: str = "data/validation_report.json") -> ValidationReport:
    """
    Run full validation suite against a raw churn DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ingested data.
    output_path : str
        Where to save the JSON validation report.

    Returns
    -------
    ValidationReport
        Contains individual rule results and overall pass/fail.

    Raises
    ------
    DataQualityError
        If any CRITICAL rule fails.
    """
    report = ValidationReport()
    n_rows = len(df)

    # ------------------------------------------------------------------
    # 1. Required columns present
    # ------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.results.append(ValidationResult(
        rule_name="required_columns_present",
        passed=len(missing_cols) == 0,
        severity=Severity.CRITICAL,
        detail=f"Missing columns: {missing_cols}" if missing_cols else "All required columns present",
    ))

    if missing_cols:
        _save_and_raise(report, output_path)

    # ------------------------------------------------------------------
    # 2. No duplicate customer IDs
    # ------------------------------------------------------------------
    n_dupes = df["customerID"].duplicated().sum()
    report.results.append(ValidationResult(
        rule_name="no_duplicate_customer_ids",
        passed=n_dupes == DUPLICATE_ID_THRESHOLD,
        severity=Severity.CRITICAL,
        detail=f"{n_dupes} duplicate customerID(s) found",
        value=float(n_dupes),
        threshold=float(DUPLICATE_ID_THRESHOLD),
    ))

    # ------------------------------------------------------------------
    # 3. Row count sanity
    # ------------------------------------------------------------------
    report.results.append(ValidationResult(
        rule_name="minimum_row_count",
        passed=n_rows >= 100,
        severity=Severity.CRITICAL,
        detail=f"Dataset has {n_rows} rows (minimum: 100)",
        value=float(n_rows),
        threshold=100.0,
    ))

    # ------------------------------------------------------------------
    # 4. Null rates
    # ------------------------------------------------------------------
    for col in ["tenure", "MonthlyCharges", "Churn", "customerID"]:
        if col not in df.columns:
            continue
        null_rate = df[col].isnull().mean()
        report.results.append(ValidationResult(
            rule_name=f"null_rate_{col}",
            passed=null_rate <= NULL_RATE_THRESHOLD,
            severity=Severity.WARNING if null_rate <= 0.20 else Severity.CRITICAL,
            detail=f"Null rate for '{col}': {null_rate:.2%}",
            value=round(null_rate, 4),
            threshold=NULL_RATE_THRESHOLD,
        ))

    # ------------------------------------------------------------------
    # 5. Numeric ranges
    # ------------------------------------------------------------------
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((numeric < lo) | (numeric > hi)).sum()
        rate = out_of_range / n_rows
        report.results.append(ValidationResult(
            rule_name=f"range_check_{col}",
            passed=rate <= 0.01,  # allow 1% tolerance
            severity=Severity.WARNING,
            detail=f"'{col}': {out_of_range} values outside [{lo}, {hi}] ({rate:.2%})",
            value=round(rate, 4),
            threshold=0.01,
        ))

    # ------------------------------------------------------------------
    # 6. Categorical values
    # ------------------------------------------------------------------
    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - allowed
        report.results.append(ValidationResult(
            rule_name=f"categorical_values_{col}",
            passed=len(unexpected) == 0,
            severity=Severity.WARNING,
            detail=(
                f"'{col}' has unexpected values: {unexpected}" if unexpected
                else f"'{col}' values are all expected"
            ),
        ))

    # ------------------------------------------------------------------
    # 7. Target distribution sanity (churn rate 10%–60%)
    # ------------------------------------------------------------------
    if "Churn" in df.columns:
        churn_col = pd.to_numeric(df["Churn"].replace({"Yes": 1, "No": 0}), errors="coerce")
        churn_rate = churn_col.mean()
        report.results.append(ValidationResult(
            rule_name="churn_rate_in_expected_range",
            passed=0.10 <= churn_rate <= 0.60,
            severity=Severity.WARNING,
            detail=f"Churn rate: {churn_rate:.2%} (expected 10%–60%)",
            value=round(float(churn_rate), 4),
        ))

    # ------------------------------------------------------------------
    # Summary logging
    # ------------------------------------------------------------------
    for r in report.results:
        level = logging.ERROR if (not r.passed and r.severity == Severity.CRITICAL) else \
                logging.WARNING if not r.passed else logging.INFO
        logger.log(level, "[%s] %s — %s", r.severity.value, r.rule_name, r.detail)

    # Save report
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Validation report saved: %s", output_path)

    if not report.passed:
        _save_and_raise(report, output_path)

    logger.info(
        "Validation PASSED — %d rules checked, %d warnings.",
        len(report.results), report.n_warnings,
    )
    return report


def _save_and_raise(report: ValidationReport, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    raise DataQualityError(
        f"Data quality validation failed — {report.n_critical_failures} critical failure(s). "
        f"See {output_path} for details."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV path to validate")
    parser.add_argument("--output", default="data/validation_report.json")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    validate(df, output_path=args.output)
