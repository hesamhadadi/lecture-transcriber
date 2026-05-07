import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field, model_validator

app = FastAPI(
    title="Lecture Transcriber API",
    description="Download one or more video URLs, extract audio, transcribe locally, and return formatted text.",
    version="0.2.0",
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
WORKDIR = Path(os.getenv("WORKDIR", BASE_DIR / "workdir"))
WORKDIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_WORKERS = max(1, int(os.getenv("TRANSCRIBE_WORKERS", "1")))

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs_lock = threading.Lock()
jobs: dict[str, "JobStatus"] = {}


class TranscribeRequest(BaseModel):
    url: str | None = None
    urls: list[str] | None = None
    language: str | None = None

    @model_validator(mode="after")
    def validate_urls(self) -> "TranscribeRequest":
        normalized_urls = self.urls or ([self.url] if self.url else [])
        normalized_urls = [url.strip() for url in normalized_urls if url and url.strip()]

        if not normalized_urls:
            raise ValueError("Provide url or urls")

        self.urls = normalized_urls
        self.url = normalized_urls[0]
        return self


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptItem(BaseModel):
    url: str
    transcript: str
    segments: list[Segment]


class TranscribeResponse(BaseModel):
    transcript: str
    chunks: int
    transcripts: list[TranscriptItem] = Field(default_factory=list)


class JobCreateResponse(BaseModel):
    job_id: str
    status_url: str
    result_url: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = 0
    phase: str = "queued"
    total_videos: int
    completed_videos: int = 0
    current_video: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    result: TranscribeResponse | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int
    phase: str
    total_videos: int
    completed_videos: int
    current_video: str | None = None
    error: str | None = None
    result_available: bool
    created_at: datetime
    updated_at: datetime


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "lecture-transcriber",
        "transcription": "local faster-whisper",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.on_event("shutdown")
def shutdown_executor() -> None:
    executor.shutdown(wait=False, cancel_futures=True)


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel(
        MODEL_NAME,
        device=MODEL_DEVICE,
        compute_type=MODEL_COMPUTE_TYPE,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def update_job(job_id: str, **updates) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return

        data = job.model_dump()
        data.update(updates)
        data["updated_at"] = utc_now()
        jobs[job_id] = JobStatus(**data)


def get_job_or_404(job_id: str) -> JobStatus:
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


def to_job_status_response(job: JobStatus) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        phase=job.phase,
        total_videos=job.total_videos,
        completed_videos=job.completed_videos,
        current_video=job.current_video,
        error=job.error,
        result_available=job.result is not None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def run_command(cmd: list[str]) -> None:
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDERR:\n{completed.stderr.strip()}"
        )


def download_with_requests(url: str, target_path: Path) -> Path:
    headers = {"User-Agent": "Mozilla/5.0"}

    with requests.get(url, stream=True, timeout=(10, 180), headers=headers) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        if "text/html" in content_type:
            raise RuntimeError("URL returned an HTML page, falling back to yt-dlp")

        with target_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    if target_path.stat().st_size == 0:
        raise RuntimeError("Downloaded file is empty")

    return target_path


def download_with_ytdlp(url: str, target_dir: Path) -> Path:
    output_template = str(target_dir / "%(id)s.%(ext)s")
    before = {path.resolve() for path in target_dir.glob("*") if path.is_file()}

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

    media_suffixes = {".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".wav"}
    media_files = [
        path
        for path in target_dir.glob("*")
        if path.is_file()
        and path.resolve() not in before
        and path.suffix.lower() in media_suffixes
    ]

    if not media_files:
        raise RuntimeError("yt-dlp did not download any media file")

    return max(media_files, key=lambda path: path.stat().st_mtime)


def download_video(url: str, job_dir: Path) -> Path:
    direct_path = job_dir / "input.mp4"

    try:
        return download_with_requests(url, direct_path)
    except Exception:
        if direct_path.exists():
            direct_path.unlink()

    return download_with_ytdlp(url, job_dir)


def extract_audio(video_path: Path, job_dir: Path) -> Path:
    audio_path = job_dir / "audio.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]

    run_command(cmd)
    return audio_path


def audio_duration_seconds(audio_path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(f"Could not read audio duration: {completed.stderr.strip()}")

    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("ffprobe returned an invalid duration") from exc

    return max(duration, 1.0)


def should_start_new_paragraph(
    paragraph_text: str,
    segment_text: str,
    gap_seconds: float,
) -> bool:
    if not paragraph_text:
        return False

    word_count = len(paragraph_text.split())
    sentence_ended = paragraph_text.rstrip().endswith((".", "?", "!"))

    return (
        gap_seconds >= 1.5
        or word_count >= 95
        or (sentence_ended and word_count >= 45)
        or (segment_text[:1].isupper() and sentence_ended and word_count >= 30)
    )


def format_segments_as_paragraphs(segments: list[Segment]) -> str:
    paragraphs: list[str] = []
    current_parts: list[str] = []
    current_text = ""
    previous_end = 0.0

    for segment in segments:
        text = " ".join(segment.text.strip().split())
        if not text:
            continue

        gap = max(0.0, segment.start - previous_end)

        if should_start_new_paragraph(current_text, text, gap):
            paragraphs.append(" ".join(current_parts).strip())
            current_parts = []
            current_text = ""

        current_parts.append(text)
        current_text = " ".join(current_parts)
        previous_end = segment.end

    if current_parts:
        paragraphs.append(" ".join(current_parts).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def transcribe_audio(
    audio_path: Path,
    language: str | None,
    progress_callback,
) -> tuple[str, list[Segment]]:
    model = get_whisper_model()
    duration = audio_duration_seconds(audio_path)
    raw_segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
    )

    segments: list[Segment] = []
    last_progress = 0

    for raw_segment in raw_segments:
        segment = Segment(
            start=round(raw_segment.start, 2),
            end=round(raw_segment.end, 2),
            text=raw_segment.text.strip(),
        )
        segments.append(segment)

        progress = min(100, int((segment.end / duration) * 100))
        if progress > last_progress:
            progress_callback(progress)
            last_progress = progress

    progress_callback(100)
    return format_segments_as_paragraphs(segments), segments


def combine_transcripts(items: list[TranscriptItem]) -> str:
    if len(items) == 1:
        return items[0].transcript

    parts = []
    for index, item in enumerate(items, start=1):
        parts.append(f"Video {index}: {item.url}\n\n{item.transcript}")

    return "\n\n---\n\n".join(parts)


def cleanup_job(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)


def process_transcription_job(job_id: str, req: TranscribeRequest) -> None:
    urls = req.urls or []
    job_dir = WORKDIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    transcripts: list[TranscriptItem] = []

    def set_video_progress(video_index: int, phase_offset: int, phase_span: int, local_progress: int) -> None:
        per_video = 100 / max(len(urls), 1)
        progress = int((video_index * per_video) + (phase_offset + phase_span * local_progress / 100) * per_video / 100)
        update_job(job_id, progress=max(0, min(progress, 99)))

    try:
        update_job(job_id, status="running", phase="starting", progress=0)

        for video_index, url in enumerate(urls):
            video_dir = job_dir / f"video_{video_index:03d}"
            video_dir.mkdir(parents=True, exist_ok=True)

            update_job(
                job_id,
                phase="downloading",
                current_video=url,
                completed_videos=video_index,
            )
            set_video_progress(video_index, 0, 20, 0)
            video_path = download_video(url, video_dir)
            set_video_progress(video_index, 0, 20, 100)

            update_job(job_id, phase="extracting_audio")
            audio_path = extract_audio(video_path, video_dir)
            set_video_progress(video_index, 20, 15, 100)

            update_job(job_id, phase="transcribing")

            def on_transcribe_progress(local_progress: int) -> None:
                set_video_progress(video_index, 35, 65, local_progress)

            transcript, segments = transcribe_audio(
                audio_path,
                language=req.language,
                progress_callback=on_transcribe_progress,
            )
            transcripts.append(
                TranscriptItem(
                    url=url,
                    transcript=transcript,
                    segments=segments,
                )
            )
            update_job(job_id, completed_videos=video_index + 1)

        response = TranscribeResponse(
            transcript=combine_transcripts(transcripts),
            chunks=len(transcripts),
            transcripts=transcripts,
        )
        update_job(
            job_id,
            status="completed",
            phase="completed",
            progress=100,
            current_video=None,
            result=response,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            phase="failed",
            error=str(exc),
            current_video=None,
        )
    finally:
        cleanup_job(job_dir)


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    job_id = str(uuid.uuid4())
    now = utc_now()

    with jobs_lock:
        jobs[job_id] = JobStatus(
            job_id=job_id,
            status="queued",
            total_videos=len(req.urls or []),
            created_at=now,
            updated_at=now,
        )

    process_transcription_job(job_id, req)
    job = get_job_or_404(job_id)

    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error)

    if not job.result:
        raise HTTPException(status_code=500, detail="Transcription did not produce a result")

    return job.result


@app.post("/transcribe/jobs", response_model=JobCreateResponse, status_code=202)
def create_transcription_job(req: TranscribeRequest):
    job_id = str(uuid.uuid4())
    now = utc_now()

    with jobs_lock:
        jobs[job_id] = JobStatus(
            job_id=job_id,
            status="queued",
            total_videos=len(req.urls or []),
            created_at=now,
            updated_at=now,
        )

    executor.submit(process_transcription_job, job_id, req)

    return JobCreateResponse(
        job_id=job_id,
        status_url=f"/transcribe/jobs/{job_id}",
        result_url=f"/transcribe/jobs/{job_id}/result",
    )


@app.get("/transcribe/jobs/{job_id}", response_model=JobStatusResponse)
def get_transcription_job(job_id: str):
    return to_job_status_response(get_job_or_404(job_id))


@app.get("/transcribe/jobs/{job_id}/result", response_model=TranscribeResponse)
def get_transcription_job_result(job_id: str):
    job = get_job_or_404(job_id)

    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error)

    if job.status != "completed" or not job.result:
        raise HTTPException(status_code=202, detail="Job is not complete yet")

    return job.result
