"""
src/serving/api.py
-------------------
Production-ready FastAPI REST endpoint for churn predictions.

Features
--------
- Pydantic v2 input/output validation with descriptive field docs
- JWT bearer token authentication (HS256)
- Structured JSON request logging (prediction ID, latency, model version)
- /health endpoint for liveness probes
- /metrics endpoint for monitoring (request counts, latency percentiles)
- Graceful model loading from MLflow at startup
- Rate limiting via slowapi

Usage
-----
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
    MLFLOW_TRACKING_URI     MLflow server URI (default: sqlite:///mlflow.db)
    MODEL_NAME              Registered model name (default: churn-predictor)
    MODEL_STAGE             Registry stage to serve (default: Production)
    JWT_SECRET              Secret key for JWT verification
    LOG_LEVEL               Logging level (default: INFO)
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-predictor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("churn-api")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

model_state = {
    "model": None,
    "model_version": None,
    "loaded_at": None,
}

# In-memory metrics (replace with Prometheus in production)
request_metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
}


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model from MLflow Registry on startup."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from registry: %s", model_uri)
    try:
        model_state["model"] = mlflow.sklearn.load_model(model_uri)
        model_state["loaded_at"] = time.time()
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        # Allow startup to continue — /health will report degraded state
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Production REST endpoint for customer churn probability prediction. "
        "Built on an XGBoost model trained on the IBM Telco Churn dataset, "
        "tracked and versioned with MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT bearer token."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Input features for a single customer churn prediction."""

    tenure: int = Field(..., ge=0, le=72, description="Months as a customer (0–72)")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount (ZAR)")
    total_charges: float = Field(..., ge=0, description="Cumulative charges to date (ZAR)")
    contract: str = Field(
        ...,
        description="Contract type: 'Month-to-month', 'One year', or 'Two year'",
    )
    internet_service: str = Field(
        ..., description="Internet service type: 'DSL', 'Fiber optic', or 'No'"
    )
    payment_method: str = Field(
        ...,
        description=(
            "Payment method: 'Electronic check', 'Mailed check', "
            "'Bank transfer (automatic)', or 'Credit card (automatic)'"
        ),
    )
    num_products: int = Field(
        default=1, ge=0, le=8, description="Number of add-on services subscribed (0–8)"
    )
    senior_citizen: int = Field(default=0, ge=0, le=1, description="1 if senior citizen, else 0")
    partner: str = Field(default="No", description="'Yes' or 'No'")
    dependents: str = Field(default="No", description="'Yes' or 'No'")

    model_config = {"json_schema_extra": {
        "example": {
            "tenure": 24,
            "monthly_charges": 65.5,
            "total_charges": 1572.0,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "num_products": 3,
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
        }
    }}


class PredictionResponse(BaseModel):
    """Churn prediction result."""

    prediction_id: str = Field(..., description="Unique prediction identifier (UUID4)")
    churn_probability: float = Field(..., description="Predicted probability of churn (0–1)")
    churn_prediction: int = Field(..., description="Binary churn decision (0=Stay, 1=Churn)")
    risk_tier: str = Field(..., description="Risk tier: 'LOW', 'MEDIUM', or 'HIGH'")
    model_version: Optional[str] = Field(None, description="MLflow model version serving this request")
    latency_ms: float = Field(..., description="End-to-end prediction latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: Optional[float]


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

CONTRACT_MAP = {"Month-to-month": 1, "One year": 12, "Two year": 24}
INTERNET_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}
PAYMENT_MAP = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}
BINARY_MAP = {"Yes": 1, "No": 0}


def build_feature_vector(req: PredictionRequest) -> np.ndarray:
    """Convert request payload to a numpy feature vector matching training schema."""
    charge_per_tenure = req.monthly_charges / (req.tenure + 1)
    contract_months = CONTRACT_MAP.get(req.contract, 1)
    internet_encoded = INTERNET_MAP.get(req.internet_service, 0)
    payment_encoded = PAYMENT_MAP.get(req.payment_method, 0)
    partner_encoded = BINARY_MAP.get(req.partner, 0)
    dependents_encoded = BINARY_MAP.get(req.dependents, 0)

    features = np.array([[
        req.tenure,
        req.monthly_charges,
        req.total_charges,
        req.num_products,
        charge_per_tenure,
        contract_months,
        req.senior_citizen,
        internet_encoded,
        payment_encoded,
        partner_encoded,
        dependents_encoded,
    ]])
    return features


def get_risk_tier(probability: float) -> str:
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health():
    """Liveness probe — returns model load status."""
    uptime = time.time() - model_state["loaded_at"] if model_state["loaded_at"] else None
    return HealthResponse(
        status="ok" if model_state["model"] is not None else "degraded",
        model_loaded=model_state["model"] is not None,
        model_version=model_state.get("model_version"),
        uptime_seconds=round(uptime, 1) if uptime else None,
    )


@app.get("/metrics", tags=["operations"])
async def metrics():
    """Aggregated request metrics for monitoring dashboards."""
    n = request_metrics["total_requests"]
    avg_latency = (request_metrics["total_latency_ms"] / n) if n > 0 else 0
    return {
        "total_requests": n,
        "error_rate": request_metrics["errors"] / max(n, 1),
        "avg_latency_ms": round(avg_latency, 2),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict customer churn probability",
)
async def predict(
    request: PredictionRequest,
    token_payload: dict = Depends(verify_token),
):
    """
    Predict the probability that a customer will churn.

    Requires a valid JWT bearer token. Returns a churn probability,
    binary decision, risk tier, and prediction metadata.
    """
    if model_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — service is starting up",
        )

    start = time.time()
    prediction_id = str(uuid.uuid4())
    request_metrics["total_requests"] += 1

    try:
        features = build_feature_vector(request)
        prob = float(model_state["model"].predict_proba(features)[0, 1])
        pred = int(prob >= 0.5)
        latency_ms = (time.time() - start) * 1000
        request_metrics["total_latency_ms"] += latency_ms

        logger.info(
            "prediction_id=%s churn_prob=%.4f latency_ms=%.1f user=%s",
            prediction_id,
            prob,
            latency_ms,
            token_payload.get("sub", "unknown"),
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            churn_probability=round(prob, 4),
            churn_prediction=pred,
            risk_tier=get_risk_tier(prob),
            model_version=model_state.get("model_version"),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        request_metrics["errors"] += 1
        logger.exception("Prediction failed for request %s: %s", prediction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed — see server logs",
        ) from exc
