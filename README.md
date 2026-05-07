# Lecture Transcriber

FastAPI service for downloading lecture videos, extracting audio with `ffmpeg`, and transcribing locally with `faster-whisper`.

The service does not require OpenAI or any external paid transcription API. All transcription runs on the host machine.

## Features

- Transcribe one video synchronously with `POST /transcribe`
- Queue one or more videos for background processing with `POST /transcribe/jobs`
- Poll reliable job progress with `GET /transcribe/jobs/{job_id}`
- Fetch completed transcripts with `GET /transcribe/jobs/{job_id}/result`
- Download direct media URLs or supported video pages through `yt-dlp`
- Format transcript segments into readable paragraphs

## Requirements

- Python 3.10+
- `ffmpeg`
- Enough disk space for temporary video and audio files

Install `ffmpeg`:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:

- API health: `http://localhost:8000/health`
- OpenAPI docs: `http://localhost:8000/docs`

## Configuration

Environment variables:

| Name | Default | Description |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `WORKDIR` | `backend/workdir` | Temporary media directory |
| `WHISPER_MODEL` | `small` | Local faster-whisper model size or path |
| `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda`, or another faster-whisper device |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute type, for example `int8`, `float16`, or `float32` |
| `TRANSCRIBE_WORKERS` | `1` | Number of background transcription workers |

## Synchronous Transcription

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/video.mp4",
    "language": "en"
  }'
```

Response:

```json
{
  "transcript": "Formatted transcript text...",
  "chunks": 1,
  "transcripts": [
    {
      "url": "https://example.com/video.mp4",
      "transcript": "Formatted transcript text...",
      "segments": [
        {
          "start": 0.0,
          "end": 4.2,
          "text": "Segment text"
        }
      ]
    }
  ]
}
```

## Background Jobs

Create a job for one or more videos:

```bash
curl -X POST http://localhost:8000/transcribe/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/lecture-1.mp4",
      "https://example.com/lecture-2.mp4"
    ],
    "language": "en"
  }'
```

Response:

```json
{
  "job_id": "4b6c8f47-2f23-4a22-9f06-76c0f3737a2a",
  "status_url": "/transcribe/jobs/4b6c8f47-2f23-4a22-9f06-76c0f3737a2a",
  "result_url": "/transcribe/jobs/4b6c8f47-2f23-4a22-9f06-76c0f3737a2a/result"
}
```

Poll progress:

```bash
curl http://localhost:8000/transcribe/jobs/4b6c8f47-2f23-4a22-9f06-76c0f3737a2a
```

Example status response:

```json
{
  "job_id": "4b6c8f47-2f23-4a22-9f06-76c0f3737a2a",
  "status": "running",
  "progress": 42,
  "phase": "transcribing",
  "total_videos": 2,
  "completed_videos": 0,
  "current_video": "https://example.com/lecture-1.mp4",
  "error": null,
  "result_available": false,
  "created_at": "2026-05-07T20:00:00Z",
  "updated_at": "2026-05-07T20:01:30Z"
}
```

Possible statuses:

- `queued`
- `running`
- `completed`
- `failed`

Fetch the result when the job is complete:

```bash
curl http://localhost:8000/transcribe/jobs/4b6c8f47-2f23-4a22-9f06-76c0f3737a2a/result
```

If the job is still running, the result endpoint returns HTTP `202`.

## Notes

- The first request may take longer because `faster-whisper` downloads or loads the selected model.
- Temporary media files are removed after each job finishes.
- Job state is kept in memory. Restarting the API clears queued and completed job records.
