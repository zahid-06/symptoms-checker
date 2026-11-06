from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "disease_model.pkl"

DISCLAIMER = (
    "Educational project using a public dataset. This is not medical advice "
    "and cannot diagnose any condition. Please consult a doctor."
)

artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    try:
        artifacts.update(joblib.load(MODEL_PATH))
        print("Model loaded.")
    except Exception as e:
        raise RuntimeError(f"Could not load model (run src/train_model.py first): {e}")
    yield
    artifacts.clear()


app = FastAPI(
    title="Symptom Checker API",
    description="Suggests likely diseases from selected symptoms. " + DISCLAIMER,
    version="2.0.0",
    lifespan=lifespan,
)


class SymptomInput(BaseModel):
    symptoms: List[str] = Field(..., min_length=2)


class Prediction(BaseModel):
    disease: str
    probability: float
    description: str
    precautions: List[str]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    severity_score: int
    max_severity: int
    disclaimer: str


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(HERE / "index.html")


@app.get("/symptoms")
def list_symptoms():
    return {"symptoms": artifacts["symptoms"]}


@app.post("/predict", response_model=PredictResponse)
def predict(data: SymptomInput):
    symptoms = artifacts["symptoms"]
    chosen = set(data.symptoms)

    unknown = sorted(chosen - set(symptoms))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown symptoms: {unknown}")

    
    X = pd.DataFrame([[1 if s in chosen else 0 for s in symptoms]], columns=symptoms)

    model = artifacts["model"]
    probabilities = model.predict_proba(X)[0]
    top = np.argsort(probabilities)[::-1][:3]

    predictions = []
    for i in top:
        if probabilities[i] < 0.01:
            continue
        disease = str(model.classes_[i])
        predictions.append(
            Prediction(
                disease=disease,
                probability=round(float(probabilities[i]), 3),
                description=artifacts["descriptions"].get(disease, "No description available."),
                precautions=artifacts["precautions"].get(disease, []),
            )
        )

    weights = artifacts["severity"]
    return PredictResponse(
        predictions=predictions,
        severity_score=sum(weights.get(s, 0) for s in chosen),
        max_severity=len(chosen) * max(weights.values()),
        disclaimer=DISCLAIMER,
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in artifacts}