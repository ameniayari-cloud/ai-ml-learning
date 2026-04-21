"""Minimal educational capstone API skeleton combining two modalities."""
from fastapi import FastAPI

app = FastAPI(title="Capstone Vision+NLP")


def predict_image_stub() -> str:
    return "image_class_placeholder"


def predict_text_stub() -> str:
    return "text_label_placeholder"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/multimodal")
def predict_multimodal() -> dict[str, str]:
    return {
        "image_prediction": predict_image_stub(),
        "text_prediction": predict_text_stub(),
    }
