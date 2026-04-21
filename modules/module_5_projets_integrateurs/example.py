"""Module 5 example: tiny FastAPI inference endpoint."""

from fastapi import FastAPI

app = FastAPI(title="ML Inference API")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict")
def predict(x: float, y: float) -> dict[str, float]:
    # Placeholder inference logic for educational purposes.
    prediction = 0.7 * x + 0.3 * y
    return {"prediction": prediction}
