"""
FastAPI service that serves predictions from the MLflow-registered
'churn-xgb-classifier' model trained by src/train.py.

Run with:
    uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import ChurnPredictionRequest, ChurnPredictionResponse

MLRUNS_DIR = Path(__file__).parent.parent / "mlruns"
MODEL_NAME = "churn-xgb-classifier"

_state = {"model": None, "model_version": None, "plan_encoding": None}


def _load_plan_encoding(run_id: str) -> dict:
    """Loads the plan-name -> encoded-int mapping logged as an artifact during training."""
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "plan_encoding.json")
    with open(local_path) as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR.resolve()}/mlflow.db")
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions found for model '{MODEL_NAME}'. "
            "Run src/train.py first."
        )
    latest = max(versions, key=lambda v: int(v.version))

    _state["model"] = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{latest.version}")
    _state["model_version"] = latest.version
    _state["plan_encoding"] = _load_plan_encoding(latest.run_id)

    print(f"Loaded {MODEL_NAME} v{_state['model_version']} (run {latest.run_id})")
    yield
    _state["model"] = None


app = FastAPI(
    title="Churn Prediction API",
    description="Serves churn predictions from the MLflow-registered XGBoost model (Phase 3).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    model = _state["model"]
    plan_encoding = _state["plan_encoding"]

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.signup_plan not in plan_encoding:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown signup_plan '{request.signup_plan}'. "
            f"Expected one of: {list(plan_encoding.keys())}",
        )

    login_count = request.login_count_30d
    tickets = request.support_tickets_30d
    ticket_to_login_ratio = tickets if login_count == 0 else round(tickets / login_count, 3)

    row = pd.DataFrame(
        [
            {
                "tenure_days": request.tenure_days,
                "monthly_charge": request.monthly_charge,
                "age": request.age,
                "login_count_30d": login_count,
                "days_since_last_login": request.days_since_last_login,
                "support_tickets_30d": tickets,
                "ticket_to_login_ratio": ticket_to_login_ratio,
                "signup_plan": plan_encoding[request.signup_plan],
            }
        ]
    )

    proba = float(model.predict_proba(row)[0, 1])
    prediction = int(proba >= 0.5)

    return ChurnPredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        model_version=str(_state["model_version"]),
    )
