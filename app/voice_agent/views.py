# views.py
from fastapi.responses import HTMLResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
import json
import tempfile
import io
from typing import AsyncGenerator
import logging
import os
from pydub import AudioSegment
from modules.stt import speech_to_text
from modules.llm import gemini_response, groq_response  # Updated to include Groq
from modules.tts import text_to_speech
from app.config import SAMPLE_RATE, WHISPER_MODEL_NAME, GROQ_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, FALLBACK_TTS

router = APIRouter(prefix="/voice_agent", tags=["Voice Agent"])

# Templates for HTML
templates = Jinja2Templates(directory="templates")

# Voice Agent Class
class VoiceAgent:
    def __init__(self, stt_mode: str = "local", tts_mode: str = "gtts", llm_mode: str = "gemini"):
        self.stt_mode = stt_mode
        self.tts_mode = tts_mode
        self.llm_mode = llm_mode
        logging.basicConfig(level=logging.INFO)

    async def process_audio_to_text(self, audio_bytes: bytes, stt_mode: str) -> str:
        """STT: Audio bytes to text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            text = speech_to_text(tmp_path, mode=stt_mode)
            logging.info(f"STT Result: {text}")
            return text if text else "[no speech]"
        finally:
            os.unlink(tmp_path)

    async def generate_response(self, text: str, llm_mode: str) -> str:
        """LLM: Text to response using Gemini or Groq."""
        if not text or text == "[no speech]":
            return "I didn't hear anything. Please speak again."
        if "bye" in text.lower() or "exit" in text.lower():
            return "Goodbye! Have a great day."
        prompt = f"""Answer the question concisely and clearly in English.

Question: {text}

Answer:"""
        if llm_mode == "gemini":
            response = gemini_response(prompt)
        elif llm_mode == "groq":
            response = groq_response(prompt)
        else:
            raise ValueError("Invalid LLM mode. Choose 'gemini' or 'groq'.")
        logging.info(f"LLM Response ({llm_mode}): {response}")
        return response

    async def text_to_audio(self, text: str, tts_mode: str) -> bytes:
        """TTS: Text to audio bytes."""
        file_path = text_to_speech(text, mode=tts_mode)
        if not file_path or not os.path.exists(file_path):
            logging.warning(f"TTS failed for {tts_mode}, falling back to {FALLBACK_TTS}")
            file_path = text_to_speech(text, mode=FALLBACK_TTS)
        if not file_path:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        os.unlink(file_path)
        return audio_bytes

# Global agent instance
agent = VoiceAgent(stt_mode="local", tts_mode="kokoro", llm_mode="gemini")

@router.get("/")
async def index(request: Request):
    """Serve HTML interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(..., description="Upload audio file (WAV/MP3)"),
    stt_mode: str = Query("local", description="STT mode: local or groq"),
    tts_mode: str = Query("kokoro", description="TTS mode: gtts, coqui, or kokoro"),
    llm_mode: str = Query("gemini", description="LLM mode: gemini or groq")
):
    """Upload audio, process speech-to-speech, return audio and text."""
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=415, detail="Only WAV/MP3 files are supported.")
    if stt_mode not in ["local", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid STT mode. Choose 'local' or 'groq'.")
    if tts_mode not in ["gtts", "coqui", "kokoro"]:
        raise HTTPException(status_code=400, detail="Invalid TTS mode. Choose 'gtts', 'coqui', or 'kokoro'.")
    if llm_mode not in ["gemini", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid LLM mode. Choose 'gemini' or 'groq'.")
    contents = await file.read()
    try:
        text = await agent.process_audio_to_text(contents, stt_mode=stt_mode)
        response_text = await agent.generate_response(text, llm_mode=llm_mode)
        audio_bytes = await agent.text_to_audio(response_text, tts_mode=tts_mode)
        return JSONResponse({
            "transcription": text,
            "response": response_text,
            "audio": audio_bytes.hex(),
            "supportMessage": {"label": "Would you like to know more?", "options": ["Record Again"]}
        })
    except Exception as e:
        logging.error(f"Upload processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.websocket("/voice-stream")
async def voice_websocket(websocket: WebSocket, stt_mode: str = "local", tts_mode: str = "kokoro", llm_mode: str = "gemini"):
    """Real-time voice streaming via WebSocket."""
    if stt_mode not in ["local", "groq"]:
        await websocket.close(code=1008, reason="Invalid STT mode. Choose 'local' or 'groq'.")
        return
    if tts_mode not in ["gtts", "coqui", "kokoro"]:
        await websocket.close(code=1008, reason="Invalid TTS mode. Choose 'gtts', 'coqui', or 'kokoro'.")
        return
    if llm_mode not in ["gemini", "groq"]:
        await websocket.close(code=1008, reason="Invalid LLM mode. Choose 'gemini' or 'groq'.")
        return
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            text = await agent.process_audio_to_text(data, stt_mode=stt_mode)
            response_text = await agent.generate_response(text, llm_mode=llm_mode)
            audio_bytes = await agent.text_to_audio(response_text, tts_mode=tts_mode)
            await websocket.send_bytes(audio_bytes)
            await websocket.send_text(json.dumps({
                "type": "complete",
                "transcription": text,
                "response": response_text,
                "supportMessage": {"label": "Would you like to know more?", "options": ["Record Again"]}
            }))
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        await websocket.send_text(json.dumps({"type": "error", "message": f"Error: {str(e)}"}))

















# # views.py
# from fastapi import APIRouter, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
# from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# import asyncio
# import json
# import tempfile
# import io
# from typing import AsyncGenerator
# import logging
# import os
# from pydub import AudioSegment
# from modules.stt import speech_to_text
# from modules.llm import gemini_response
# from modules.tts import text_to_speech
# from config import SAMPLE_RATE, WHISPER_MODEL_NAME, GROQ_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, FALLBACK_TTS

# router = APIRouter(prefix="/voice_agent", tags=["Voice Agent"])

# # Templates for HTML
# templates = Jinja2Templates(directory="templates")

# # Voice Agent Class
# class VoiceAgent:
#     def __init__(self, stt_mode: str = "local", tts_mode: str = "gtts"):
#         self.stt_mode = stt_mode
#         self.tts_mode = tts_mode
#         logging.basicConfig(level=logging.INFO)

#     async def process_audio_to_text(self, audio_bytes: bytes) -> str:
#         """STT: Audio bytes to text."""
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#             tmp.write(audio_bytes)
#             tmp_path = tmp.name
#         try:
#             text = speech_to_text(tmp_path, mode=self.stt_mode)
#             logging.info(f"STT Result: {text}")
#             return text if text else "[no speech]"
#         finally:
#             os.unlink(tmp_path)

#     async def generate_response(self, text: str) -> str:
#         """LLM: Text to response, prioritizing Hindi if possible."""
#         if not text or text == "[no speech]":
#             return "मैंने कुछ सुना नहीं। कृपया फिर से बोलें।"  # Hindi priority
#         if "bye" in text.lower() or "exit" in text.lower():
#             return "अलविदा! आपका दिन शुभ हो।"  # Hindi priority
#         prompt = f"""प्रश्न का उत्तर हिंदी में दें यदि संभव हो, अन्यथा अंग्रेजी में। संक्षिप्त और स्पष्ट उत्तर प्रदान करें।

# Question: {text}

# Answer:"""
#         response = gemini_response(prompt)
#         logging.info(f"LLM Response: {response}")
#         return response

#     async def text_to_audio(self, text: str) -> bytes:
#         """TTS: Text to audio bytes."""
#         file_path = text_to_speech(text, mode=self.tts_mode)
#         if not file_path or not os.path.exists(file_path):
#             logging.warning(f"TTS failed for {self.tts_mode}, falling back to {FALLBACK_TTS}")
#             file_path = text_to_speech(text, mode=FALLBACK_TTS)
#         if not file_path:
#             raise HTTPException(status_code=500, detail="TTS generation failed")
#         with open(file_path, "rb") as f:
#             audio_bytes = f.read()
#         os.unlink(file_path)
#         return audio_bytes

#     async def stream_llm_response(self, query: str) -> AsyncGenerator[str, None]:
#         """Stream LLM response chunks (text)."""
#         full_response = await self.generate_response(query)
#         chunk_size = 10
#         for i in range(0, len(full_response), chunk_size):
#             chunk = full_response[i:i + chunk_size]
#             yield chunk
#             await asyncio.sleep(0.05)

# # Global agent instance
# agent = VoiceAgent(stt_mode="local", tts_mode="kokoro")

# @router.get("/")
# async def index(request: Request):
#     """Serve HTML interface."""
#     return templates.TemplateResponse("index.html", {"request": request})

# @router.post("/upload")
# async def upload_audio(file: UploadFile = File(..., description="Upload audio file (WAV/MP3)")):
#     """Upload audio, process speech-to-speech, return TTS audio."""
#     if not file.filename.lower().endswith(('.wav', '.mp3')):
#         raise HTTPException(status_code=415, detail="केवल WAV/MP3 फ़ाइलें समर्थित हैं।")
#     contents = await file.read()
#     try:
#         text = await agent.process_audio_to_text(contents)
#         response_text = await agent.generate_response(text)
#         audio_bytes = await agent.text_to_audio(response_text)
#         return StreamingResponse(
#             io.BytesIO(audio_bytes),
#             media_type="audio/mp3",
#             headers={"Content-Disposition": "attachment; filename=response.mp3"}
#         )
#     except Exception as e:
#         logging.error(f"Upload processing failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"प्रोसेसिंग विफल: {str(e)}")

# @router.get("/stream")
# async def voice_stream(
#     query: str = Query(..., description="Text query (or transcribed speech, supports Hindi/English)"),
#     top_k: int = Query(5, ge=1, le=20, description="Not used for voice, kept for compatibility")
# ):
#     """Stream LLM response (text chunks), then full TTS audio. Bilingual support."""
#     async def event_stream() -> AsyncGenerator[str, None]:
#         try:
#             async for chunk in agent.stream_llm_response(query):
#                 yield f"data: {json.dumps({'type': 'text_chunk', 'chunk': chunk})}\n\n"
#             full_response = await agent.generate_response(query)
#             audio_bytes = await agent.text_to_audio(full_response)
#             yield f"data: {json.dumps({'type': 'audio', 'data': audio_bytes.hex()})}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#             yield f"data: {json.dumps({'supportMessage': {'label': 'क्या आप और जानना चाहेंगे?', 'options': ['Text Query', 'Upload Audio', 'Stream Again']}})}\n\n"
#         except Exception as e:
#             logging.error(f"Stream error: {str(e)}")
#             yield f"data: {json.dumps({'type': 'error', 'message': f'त्रुटि: {str(e)}'})}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"

#     return StreamingResponse(event_stream(), media_type="text/event-stream")

# @router.get("/query")
# async def voice_query(
#     query: str = Query(..., description="Text query (supports Hindi/English)"),
#     include_llm_response: bool = Query(True, description="Include LLM response"),
#     include_tts: bool = Query(True, description="Include TTS audio")
# ):
#     """Text query: LLM response + optional TTS audio."""
#     try:
#         response_text = await agent.generate_response(query) if include_llm_response else ""
#         data = {"query": query, "response": response_text}
#         if include_tts:
#             audio_bytes = await agent.text_to_audio(response_text)
#             with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
#                 tmp.write(audio_bytes)
#                 tmp_name = os.path.basename(tmp.name)
#                 os.rename(tmp.name, f"temp_audio/{tmp_name}")
#                 data["tts_url"] = f"/temp_audio/{tmp_name}"
#         return JSONResponse(data)
#     except Exception as e:
#         logging.error(f"Query failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"क्वेरी विफल: {str(e)}")

# @router.websocket("/voice-stream")
# async def voice_websocket(websocket: WebSocket):
#     """Real-time voice streaming via WebSocket."""
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_bytes()
#             text = await agent.process_audio_to_text(data)
#             response_text = await agent.generate_response(text)
#             audio_bytes = await agent.text_to_audio(response_text)
#             await websocket.send_bytes(audio_bytes)
#             await websocket.send_text(json.dumps({
#                 "type": "complete",
#                 "data": {"message": response_text, "supportMessage": {"label": "और?", "options": ["Text Query", "Upload Audio"]}}
#             }))
#     except WebSocketDisconnect:
#         logging.info("WebSocket disconnected")
#     except Exception as e:
#         logging.error(f"WebSocket error: {str(e)}")
#         await websocket.send_text(json.dumps({"type": "error", "message": f"त्रुटि: {str(e)}"}))