import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from pydub import AudioSegment

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="Lecture Transcriber API",
    description="Download a video URL, extract audio, transcribe it, and return text.",
    version="0.1.0",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
WORKDIR = BASE_DIR / "workdir"
WORKDIR.mkdir(exist_ok=True)


class TranscribeRequest(BaseModel):
    url: str
    language: str | None = None


class TranscribeResponse(BaseModel):
    transcript: str
    chunks: int


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "lecture-transcriber",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def run_command(cmd: List[str]) -> None:
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDERR:\n{completed.stderr}"
        )


def download_with_requests(url: str, target_path: Path) -> Path:
    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    with requests.get(url, stream=True, timeout=180, headers=headers) as r:
        r.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return target_path


def download_with_ytdlp(url: str, target_dir: Path) -> Path:
    output_template = str(target_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url,
    ]

    run_command(cmd)

    files = list(target_dir.glob("*"))
    media_files = [
        f for f in files
        if f.is_file() and f.suffix.lower() in [".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3"]
    ]

    if not media_files:
        raise RuntimeError("yt-dlp did not download any media file")

    return media_files[0]


def download_video(url: str, job_dir: Path) -> Path:
    direct_path = job_dir / "input.mp4"

    try:
        return download_with_requests(url, direct_path)
    except Exception:
        if direct_path.exists():
            direct_path.unlink(missing_ok=True)

    return download_with_ytdlp(url, job_dir)


def extract_audio(video_path: Path, job_dir: Path) -> Path:
    audio_path = job_dir / "audio.mp3"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-b:a",
        "64k",
        str(audio_path),
    ]

    run_command(cmd)
    return audio_path


def split_audio(audio_path: Path, job_dir: Path, chunk_minutes: int = 8) -> List[Path]:
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks: List[Path] = []

    for index, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = job_dir / f"chunk_{index:03d}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate="64k")
        chunks.append(chunk_path)

    return chunks


def transcribe_chunk(chunk_path: Path, language: str | None = None) -> str:
    kwargs = {
        "model": "gpt-4o-mini-transcribe",
        "file": open(chunk_path, "rb"),
    }

    if language:
        kwargs["language"] = language

    try:
        result = client.audio.transcriptions.create(**kwargs)
        return result.text
    finally:
        kwargs["file"].close()


def cleanup_job(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    job_id = str(uuid.uuid4())
    job_dir = WORKDIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = download_video(req.url, job_dir)
        audio_path = extract_audio(video_path, job_dir)
        chunks = split_audio(audio_path, job_dir, chunk_minutes=8)

        transcript_parts = []

        for i, chunk_path in enumerate(chunks, start=1):
            text = transcribe_chunk(chunk_path, language=req.language)
            transcript_parts.append(f"[Chunk {i}/{len(chunks)}]\n{text}")

        transcript = "\n\n".join(transcript_parts)

        return TranscribeResponse(
            transcript=transcript,
            chunks=len(chunks),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cleanup_job(job_dir)
