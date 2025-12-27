from pydantic import BaseModel

class JobPostRequest(BaseModel):
    title: str
    description: str
    requirements: str = ""
    company_profile: str = ""
    benefits: str = ""
    telecommuting: int
    has_company_logo: int
    has_questions: int

class PredictionResponse(BaseModel):
    label: str
    confidence: float
