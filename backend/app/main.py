from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import run_agent_turn

app = FastAPI(title="Aster & Row Support Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/")
def read_root():
    return {"status": "ok", "service": "aster-row-support-agent"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = run_agent_turn(request.message, request.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="The support agent is temporarily unavailable. Please try again shortly.",
        ) from None
    return response.model_dump()
