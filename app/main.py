from fastapi import FastAPI
import joblib
import pandas as pd

import os
import csv
from datetime import datetime

from app.schema import JobPostRequest, PredictionResponse
from app.config import MODEL_PATH, THRESHOLD, LOG_FILE

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

    # Log entry
    log_prediction(
        {
            "full_text": full_text,
            "telecommuting": job.telecommuting,
            "has_company_logo": job.has_company_logo,
            "has_questions": job.has_questions,
        },
        label,
        float(round(prob, 3))
    )

    return PredictionResponse(
        label=label,
        confidence=round(prob, 3)
    )

def log_prediction(data: dict, label: str, confidence: float):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "full_text",
                "telecommuting",
                "has_company_logo",
                "has_questions",
                "predicted_label",
                "confidence"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            data["full_text"],
            data["telecommuting"],
            data["has_company_logo"],
            data["has_questions"],
            label,
            confidence
        ])
