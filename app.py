import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, List

app = FastAPI(title="UDD M7 Predictor", version="1.0")

model = joblib.load("modelo_udds_m7.joblib")

class PredictRequest(BaseModel):
    record: Dict[str, Any] = Field(..., description="Un registro con las columnas del modelo")

class PredictBatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Lista de registros")

@app.get("/")
def root():
    return {"status": "ok", "message": "UDD M7 Predictor API"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.record])
    pred = model.predict(df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        classes = list(model.classes_)
        proba = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
    return {"prediction": str(pred), "probabilities": proba}

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    df = pd.DataFrame(req.records)
    preds = model.predict(df)
    response = {"predictions": [str(p) for p in preds]}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)
        classes = list(model.classes_)
        response["probabilities"] = [
            {str(c): round(float(p), 4) for c, p in zip(classes, row)} for row in probs
        ]
    return response
