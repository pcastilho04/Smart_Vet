from typing import Any

from huggingface_hub import InferenceClient


class HuggingFaceService:
    def __init__(self, token: str, symptoms_model_id: str, image_model_id: str) -> None:
        self.client = InferenceClient(token=token or None)
        self.symptoms_model_id = symptoms_model_id
        self.image_model_id = image_model_id

    @staticmethod
    def _normalize_predictions(result: Any) -> list[dict[str, Any]]:
        if result is None:
            return []

        if isinstance(result, dict):
            result = [result]

        # Some endpoints can return nested arrays for batches.
        if isinstance(result, list) and result and isinstance(result[0], list):
            flattened: list[Any] = []
            for batch_item in result:
                if isinstance(batch_item, list):
                    flattened.extend(batch_item)
                else:
                    flattened.append(batch_item)
            result = flattened

        normalized: list[dict[str, Any]] = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    normalized.append(
                        {
                            "label": str(item.get("label", "unknown")),
                            "score": float(item.get("score")) if item.get("score") is not None else None,
                            "raw": item,
                        }
                    )
                else:
                    # Handles SDK objects like ClassificationOutput that expose attributes.
                    label = getattr(item, "label", "unknown")
                    score = getattr(item, "score", None)
                    normalized.append(
                        {
                            "label": str(label),
                            "score": float(score) if score is not None else None,
                            "raw": {"value": str(item)},
                        }
                    )

        normalized.sort(key=lambda pred: pred.get("score") or 0.0, reverse=True)

        return normalized

    def predict_symptoms(self, text: str) -> list[dict[str, Any]]:
        if not self.symptoms_model_id:
            raise ValueError("HF_SYMPTOMS_MODEL_ID is not configured")

        result = self.client.text_classification(text=text, model=self.symptoms_model_id)
        return self._normalize_predictions(result)

    def predict_image(self, image_bytes: bytes) -> list[dict[str, Any]]:
        if not self.image_model_id:
            raise ValueError("HF_IMAGE_MODEL_ID is not configured")

        result = self.client.image_classification(image=image_bytes, model=self.image_model_id)
        return self._normalize_predictions(result)
