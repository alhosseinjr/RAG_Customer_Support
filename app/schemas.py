from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Customer's message")
    session_id: str | None = Field(default=None, description="Optional session/conversation id")


class ChatResponse(BaseModel):
    response: str
    detected_language: str
    language_confidence: float
    sentiment: str
    sentiment_confidence: float
    intent: str
    escalate: bool
    used_rag: bool
    retrieved_categories: list[str]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
