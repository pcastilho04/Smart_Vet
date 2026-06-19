import pickle
from functools import lru_cache

import pandas as pd
from huggingface_hub import hf_hub_download

from app.config import Settings
from app.schemas import ConditionPredictionRequest, ConditionPredictionResponse


def normalize_text(value: str | None) -> str:
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


def build_features(payload: ConditionPredictionRequest) -> pd.DataFrame:
    symptoms = [
        normalize_text(payload.symptoms1),
        normalize_text(payload.symptoms2),
        normalize_text(payload.symptoms3),
        normalize_text(payload.symptoms4),
        normalize_text(payload.symptoms5),
    ]

    row = {
        "AnimalName": normalize_text(payload.animal_name),
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


class TabularModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @lru_cache(maxsize=1)
    def load_model(self):
        if not self.settings.hf_model_repo:
            raise ValueError("HF_MODEL_REPO is not configured")

        model_path = hf_hub_download(
            repo_id=self.settings.hf_model_repo,
            filename=self.settings.hf_model_filename,
            token=self.settings.hf_token or None,
        )
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)

    def predict(self, payload: ConditionPredictionRequest) -> ConditionPredictionResponse:
        model = self.load_model()
        features = build_features(payload)

        prediction = int(model.predict(features)[0])
        probability_yes = None

        if hasattr(model, "predict_proba"):
            classes = [str(class_item) for class_item in getattr(model, "classes_", [])]
            positive_index = 1 if len(classes) <= 1 else (classes.index("1") if "1" in classes else 1)
            probability_yes = float(model.predict_proba(features)[0][positive_index])

        return ConditionPredictionResponse(
            prediction_binary=prediction,
            prediction_label="Yes - perigoso" if prediction == 1 else "No - nao perigoso",
            probability_yes=probability_yes,
            input_used=features.iloc[0].to_dict(),
        )
