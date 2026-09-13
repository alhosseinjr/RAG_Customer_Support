"""
FastAPI deployment for the RAG customer support chatbot.

Run locally (after all four modules are trained and the index is built):
    uvicorn app.main:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "Where is my order?"}'
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from src.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

_state = {"models_loaded": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up all four models once at startup instead of on the first
    # request, so latency is consistent and a missing artifact fails fast
    # with a clear error instead of on a live customer request.
    try:
        from src.intent.predict import _load_pipeline as _load_intent
        from src.language_detection.predict import _load_pipeline as _load_lang
        from src.rag.retriever import _load_index, _load_embedder
        from src.sentiment.predict import _load_model as _load_sentiment

        _load_lang()
        _load_sentiment()
        _load_intent()
        _load_index()
        _load_embedder()
        _state["models_loaded"] = True
        log.info("All models loaded successfully.")
    except FileNotFoundError as e:
        log.warning("Startup without full model set: %s", e)
        log.warning("The /chat endpoint will fail until all modules are trained.")
    yield


app = FastAPI(
    title="E-commerce Customer Support Chatbot",
    description="RAG-based support chatbot: language detection -> sentiment -> intent -> grounded generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", models_loaded=_state["models_loaded"])


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not _state["models_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Models are not fully trained/loaded yet. Run scripts/run_full_pipeline.py first.",
        )
    try:
        result = run_pipeline(req.message)
    except Exception as e:  # noqa: BLE001 - surface a clean 500 instead of leaking internals
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail="Something went wrong processing that message.") from e

    return ChatResponse(
        response=result.response,
        detected_language=result.detected_language,
        language_confidence=result.language_confidence,
        sentiment=result.sentiment,
        sentiment_confidence=result.sentiment_confidence,
        intent=result.intent,
        escalate=result.escalate,
        used_rag=result.used_rag,
        low_confidence_retrieval=result.low_confidence_retrieval,
        retrieved_categories=result.retrieved_categories,
    )


# Mounted last and at "/" so it only catches requests that don't match
# /chat or /health above -- serves the chat UI at http://localhost:8000/.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
