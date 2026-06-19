import os
import pickle
from functools import lru_cache
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field


HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "guicon/techchallenge-animal-condition-model")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_model.pkl")
HF_TOKEN = os.getenv("HF_TOKEN")


app = FastAPI(
    title="Tech Challenge Animal Condition Inference API",
    description="API de inferencia que baixa o modelo do Hugging Face e executa predict.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    AnimalName: str = Field(..., example="dog")
    symptoms1: str = Field(..., example="fever")
    symptoms2: str = Field(..., example="diarrhea")
    symptoms3: str = Field(..., example="vomiting")
    symptoms4: str = Field(..., example="dehydration")
    symptoms5: str = Field(..., example="pains")


class PredictionResponse(BaseModel):
    prediction_binary: int
    prediction_label: str
    probability_yes: Optional[float]
    input_used: dict


def normalize_text(value):
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    text = " ".join(text.split())
    replacements = {
        "seizuers": "seizures",
        "anorexia": "loss of appetite",
        "poor appetite": "loss of appetite",
        "tiredness": "fatigue",
    }
    return replacements.get(text, text)


def build_features(payload: PredictionRequest) -> pd.DataFrame:
    animal = normalize_text(payload.AnimalName)
    symptoms: List[str] = [
        normalize_text(payload.symptoms1),
        normalize_text(payload.symptoms2),
        normalize_text(payload.symptoms3),
        normalize_text(payload.symptoms4),
        normalize_text(payload.symptoms5),
    ]

    row = {
        "AnimalName": animal,
        "symptoms1": symptoms[0],
        "symptoms2": symptoms[1],
        "symptoms3": symptoms[2],
        "symptoms4": symptoms[3],
        "symptoms5": symptoms[4],
        "unique_symptom_count": len(set(symptoms)),
        "unknown_symptom_count": sum(symptom == "unknown" for symptom in symptoms),
        "symptom_text_length": len(" ".join(symptoms)),
    }
    return pd.DataFrame([row])


@lru_cache(maxsize=1)
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILENAME,
        token=HF_TOKEN,
    )
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


@app.get("/")
def healthcheck():
    return {
        "status": "ok",
        "message": "API de inferencia ativa.",
        "model_repo": HF_MODEL_REPO,
        "model_file": HF_MODEL_FILENAME,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    model = load_model()
    features = build_features(payload)

    prediction = int(model.predict(features)[0])
    probability_yes = None

    if hasattr(model, "predict_proba"):
        probability_yes = float(model.predict_proba(features)[0][1])

    return PredictionResponse(
        prediction_binary=prediction,
        prediction_label="Yes - perigoso" if prediction == 1 else "No - nao perigoso",
        probability_yes=probability_yes,
        input_used=features.iloc[0].to_dict(),
    )
