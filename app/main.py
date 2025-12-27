from fastapi import FastAPI
import joblib
import pandas as pd

from app.schema import JobPostRequest, PredictionResponse
from app.config import MODEL_PATH, THRESHOLD

app = FastAPI(title="Fake Job Detection API")

# Load model at startup
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(job: JobPostRequest):

    full_text = (
        job.title + " " +
        job.company_profile + " " +
        job.description + " " +
        job.requirements + " " +
        job.benefits
    )

    text_length = len(full_text)

    input_df = pd.DataFrame([{
        "full_text": full_text,
        "telecommuting": job.telecommuting,
        "has_company_logo": job.has_company_logo,
        "has_questions": job.has_questions,
        "text_length": text_length
    }])

    prob = model.predict_proba(input_df)[0][1]

    label = "Fake" if prob >= THRESHOLD else "Real"

    return PredictionResponse(
        label=label,
        confidence=round(prob, 3)
    )
