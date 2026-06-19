from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.hf_client import HuggingFaceService
from app.schemas import (
    CombinedPredictionResponse,
    ConditionPredictionRequest,
    ConditionPredictionResponse,
    InferenceItem,
    PredictionResponse,
    SymptomsRequest,
)
from app.tabular_model import TabularModelService

settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hf_service = HuggingFaceService(
    token=settings.hf_token,
    symptoms_model_id=settings.hf_symptoms_model_id,
    image_model_id=settings.hf_image_model_id,
)
tabular_service = TabularModelService(settings)


def build_prediction_response(source: str, model_id: str, predictions: list[dict]) -> PredictionResponse:
    top_label = predictions[0]["label"] if predictions else None
    top_score = predictions[0]["score"] if predictions else None

    triage_level = "indeterminado"
    recommendation = "Sem predicoes suficientes para orientar triagem."

    if top_label and top_score is not None:
        label_is_positive = top_label.lower() in settings.positive_labels
        if label_is_positive and top_score >= settings.triage_high_threshold:
            triage_level = "alto"
            recommendation = "Encaminhar para atendimento veterinario imediato."
        elif label_is_positive and top_score >= settings.triage_medium_threshold:
            triage_level = "medio"
            recommendation = "Avaliar em consulta veterinaria nas proximas horas."
        else:
            triage_level = "baixo"
            recommendation = "Monitorar sinais e manter acompanhamento veterinario de rotina."

    return PredictionResponse(
        source=source,
        model_id=model_id,
        predictions=[InferenceItem(**item) for item in predictions],
        top_label=top_label,
        top_score=top_score,
        triage_level=triage_level,
        recommendation=recommendation,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.api_env}


@app.post("/predict/symptoms", response_model=PredictionResponse)
def predict_symptoms(payload: SymptomsRequest) -> PredictionResponse:
    try:
        predictions = hf_service.predict_symptoms(payload.symptoms)
        return build_prediction_response("symptoms", settings.hf_symptoms_model_id, predictions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Symptoms inference failed: {exc}") from exc


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    try:
        image_bytes = await file.read()
        predictions = hf_service.predict_image(image_bytes)
        return build_prediction_response("image", settings.hf_image_model_id, predictions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image inference failed: {exc}") from exc


@app.post("/predict", response_model=CombinedPredictionResponse)
async def predict(
    symptoms: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
) -> CombinedPredictionResponse:
    if not symptoms and not image:
        raise HTTPException(status_code=400, detail="Provide symptoms text, image, or both")

    symptoms_result = None
    image_result = None

    if symptoms:
        predictions = hf_service.predict_symptoms(symptoms)
        symptoms_result = build_prediction_response("symptoms", settings.hf_symptoms_model_id, predictions)

    if image:
        image_bytes = await image.read()
        predictions = hf_service.predict_image(image_bytes)
        image_result = build_prediction_response("image", settings.hf_image_model_id, predictions)

    return CombinedPredictionResponse(symptoms_result=symptoms_result, image_result=image_result)


@app.post("/predict/condition", response_model=ConditionPredictionResponse)
def predict_condition(payload: ConditionPredictionRequest) -> ConditionPredictionResponse:
    try:
        return tabular_service.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Condition inference failed: {exc}") from exc
