# main.py
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.voice_agent.views import router as voice_router
# from app.livekit.views import router as livekit_router

app = FastAPI(title="Voice Agent API", description="Speech-to-Speech API with Streaming")

#ross
# CORS for browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates for HTML
templates = Jinja2Templates(directory="templates")

# Include voice agent router
app.include_router(voice_router)
# app.include_router(livekit_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)