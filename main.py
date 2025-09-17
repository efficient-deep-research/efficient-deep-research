import json
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    query: str
    iid: str


class EvaluateResponse(BaseModel):
    query_id: str
    generated_response: str


class RunRequest(BaseModel):
    query: str


app = FastAPI()


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    return EvaluateResponse(query_id=request.iid, generated_response=f"Response to query: {request.query}")


def fake_response_streamer() -> Iterator[str]:
    items = [
        {"intermediate_steps": "...", "final_report": None, "is_intermediate": True, "complete": False},
        {"intermediate_steps": "...|||---|||...", "final_report": "...", "is_intermediate": False, "complete": False},
        {"intermediate_steps": "...|||---|||...", "final_report": "...", "is_intermediate": False, "complete": True},
    ]

    for item in items:
        yield f"data: {json.dumps(item)}\n"


@app.post("/run")
def run(request: RunRequest) -> StreamingResponse:
    return StreamingResponse(fake_response_streamer(), media_type="text/event-stream")
