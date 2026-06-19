from typing import Any

from pydantic import BaseModel, Field


class SymptomsRequest(BaseModel):
    symptoms: str = Field(..., min_length=3, description="Symptoms reported by the pet owner")


class InferenceItem(BaseModel):
    label: str
    score: float | None = None
    raw: dict[str, Any] | None = None


class PredictionResponse(BaseModel):
    source: str
    model_id: str
    predictions: list[InferenceItem]
    top_label: str | None = None
    top_score: float | None = None
    triage_level: str = "indeterminado"
    recommendation: str


class CombinedPredictionResponse(BaseModel):
    symptoms_result: PredictionResponse | None = None
    image_result: PredictionResponse | None = None


class ConditionPredictionRequest(BaseModel):
    animal_name: str = Field(..., min_length=2, description="Animal name/species", examples=["dog"])
    symptoms1: str = Field(..., min_length=1, examples=["fever"])
    symptoms2: str = Field(..., min_length=1, examples=["diarrhea"])
    symptoms3: str = Field(..., min_length=1, examples=["vomiting"])
    symptoms4: str = Field(..., min_length=1, examples=["dehydration"])
    symptoms5: str = Field(..., min_length=1, examples=["pains"])


class ConditionPredictionResponse(BaseModel):
    prediction_binary: int
    prediction_label: str
    probability_yes: float | None = None
    input_used: dict[str, Any]
