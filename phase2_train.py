"""
Trains an XGBoost classifier on the churn feature table produced by Phase 1,
evaluates it, and saves the model + a SHAP explainer so the Streamlit
dashboard (app.py) can load both without retraining.

Usage:
    python src/train.py
"""
import json
from pathlib import Path

import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).parent.parent / "data" / "customer_churn_features.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "signup_plan",
    "tenure_days",
    "monthly_charge",
    "age",
    "login_count_30d",
    "days_since_last_login",
    "support_tickets_30d",
    "ticket_to_login_ratio",
]
TARGET_COL = "churned"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run phase1-sql-feature-pipelines/run.sh first "
            "to generate the feature table."
        )
    return pd.read_csv(DATA_PATH)


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df["signup_plan_encoded"] = le.fit_transform(df["signup_plan"])

    feature_cols_encoded = [c for c in FEATURE_COLS if c != "signup_plan"] + ["signup_plan_encoded"]
    X = df[feature_cols_encoded].rename(columns={"signup_plan_encoded": "signup_plan"})
    y = df[TARGET_COL]
    return X, y, le


def main():
    df = load_data()
    X, y, plan_encoder = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    print(f"Test AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

    # Save model
    model.save_model(MODEL_DIR / "xgb_churn_model.json")

    # Save plan label encoding so the dashboard can map plan names <-> encoded values
    plan_mapping = {cls: int(idx) for idx, cls in enumerate(plan_encoder.classes_)}
    with open(MODEL_DIR / "plan_encoding.json", "w") as f:
        json.dump(plan_mapping, f, indent=2)

    # Save metrics
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"test_auc": auc, "classification_report": report}, f, indent=2)

    # Save a sample of the test set (with true labels) for the dashboard to explore
    X_test.assign(churned=y_test.values).to_csv(MODEL_DIR / "test_sample.csv", index=False)

    # Precompute SHAP values on the test set and save them for fast dashboard loading
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    pd.DataFrame(shap_values, columns=X_test.columns).to_csv(
        MODEL_DIR / "shap_values_test.csv", index=False
    )

    print(f"\nModel, metrics, and SHAP values saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
