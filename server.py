"""
server.py — FastAPI web server for the OpsCortex chat UI.

Serves a single-page chat interface and a streaming chat endpoint. Anyone who
can reach the server can converse with the OpsCortex agent in the browser, run
the pipeline, and approve/deny high-risk plans.

Run:
    python server.py
    # then open http://127.0.0.1:8000

Requires ANTHROPIC_API_KEY + OPSCORTEX_LLM_BACKEND=anthropic (loaded from .env).

NOTE: this dev server has no authentication. Do not expose it to the public
internet as-is — put it behind auth / a reverse proxy first.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import config
import web_agent

log = config.get_logger("server")
app = FastAPI(title="OpsCortex Chat")

_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_WEB_DIR, "index.html"))


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "backend": config.LLM_BACKEND,
        "synthesis_model": config.SYNTHESIS_MODEL,
        "triage_model": config.TRIAGE_MODEL,
        "key_present": bool(config.ANTHROPIC_API_KEY),
    })


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    session = web_agent.get_session(body.get("session_id"))
    user_text = (body.get("message") or "").strip()

    def gen():
        if not user_text:
            yield web_agent._sse("error", {"message": "empty message"})
            return
        yield from web_agent.stream_agent(session, user_text)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _preflight() -> None:
    if config.LLM_BACKEND != "anthropic" or not config.ANTHROPIC_API_KEY:
        log.warning("The web agent needs the real Anthropic backend. Set "
                    "OPSCORTEX_LLM_BACKEND=anthropic and ANTHROPIC_API_KEY "
                    "(e.g. in .env). Current backend=%s, key_present=%s",
                    config.LLM_BACKEND, bool(config.ANTHROPIC_API_KEY))


if __name__ == "__main__":
    import uvicorn
    _preflight()
    print("OpsCortex chat UI -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
