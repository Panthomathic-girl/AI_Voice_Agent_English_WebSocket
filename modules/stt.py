# modules/stt.py
import whisper
import os
from groq import Groq
from app.config import WHISPER_MODEL_NAME, GROQ_API_KEY

# Local Whisper model (loaded on demand)
local_model = None

def load_local_whisper():
    global local_model
    if local_model is None:
        print("Loading Local Whisper model... (one-time)")
        local_model = whisper.load_model(WHISPER_MODEL_NAME)
        print(f"Local Whisper '{WHISPER_MODEL_NAME}' loaded.")
    return local_model

def local_speech_to_text(audio_file):
    if not audio_file or not os.path.exists(audio_file):
        return ""
    try:
        model = load_local_whisper()
        result = model.transcribe(audio_file, language="en", fp16=False)
        text = result["text"].strip()
        print(f"Local STT: '{text}'")
        return text
    except Exception as e:
        print(f"Local STT Error: {e}")
        return ""

def groq_speech_to_text(audio_file):
    if not audio_file or not os.path.exists(audio_file):
        return ""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio_file, f.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en"
            )
        text = transcription.strip()
        print(f"Groq STT: '{text}'")
        return text
    except Exception as e:
        print(f"Groq STT Error: {e}")
        return ""

def speech_to_text(audio_file, mode="local"):
    if mode == "local":
        return local_speech_to_text(audio_file)
    elif mode == "groq":
        return groq_speech_to_text(audio_file)
    else:
        raise ValueError("Invalid STT mode. Choose 'local' or 'groq'.")