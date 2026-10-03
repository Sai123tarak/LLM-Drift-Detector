from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

try:
    from llm_drift_detector.detector import LLMDriftDetector
except ModuleNotFoundError:  # pragma: no cover - compatibility with local execution
    from detector import LLMDriftDetector


app = FastAPI(title="LLM Drift Detector API")
detector = LLMDriftDetector()

# Simulating basic memory-based logging storage for simplicity
# Using a shared in-memory dataset makes the dashboard and API work together.
datastore = {
    "reference_queries": [
        "What is machine learning?",
        "How to train a neural network?",
        "Define deep learning.",
    ],
    "target_queries": [],
}


class QueryLog(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty.")
        return cleaned


@app.get("/")
def read_root():
    return {"status": "running", "message": "LLM Drift Detector API is active."}


@app.post("/log_query")
def log_query(payload: QueryLog):
    datastore["target_queries"].append(payload.text)
    return {"status": "logged", "current_count": len(datastore["target_queries"])}


@app.get("/calculate_drift")
def calculate_drift():
    target_queries = [query.strip() for query in datastore["target_queries"] if query and query.strip()]
    reference_queries = [query.strip() for query in datastore["reference_queries"] if query and query.strip()]

    if not target_queries:
        return {"drift_score": 0.0, "status": "insufficient_data"}

    if not reference_queries:
        raise HTTPException(status_code=400, detail="Reference dataset is empty.")

    ref_embs = detector.get_embeddings(reference_queries)
    tar_embs = detector.get_embeddings(target_queries)
    drift_score = detector.calculate_drift(ref_embs, tar_embs)
    return {"drift_score": drift_score, "status": "success"}
