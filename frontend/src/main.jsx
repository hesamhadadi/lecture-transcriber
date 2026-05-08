import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  CheckCircle2,
  Clipboard,
  Download,
  FileText,
  Gauge,
  HardDrive,
  Info,
  Link,
  Loader2,
  Play,
  RotateCcw,
  Server,
  Sparkles,
  Trash2,
  Video
} from "lucide-react";
import "./styles.css";

const STORAGE_KEY = "lecture-transcriber-jobs";
const MAX_POLL_ERRORS = 6;

function loadJobs() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveJobs(jobs) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs));
}

function formatDuration(seconds) {
  if (!seconds) return "Unknown duration";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const remaining = String(total % 60).padStart(2, "0");
  return `${minutes}:${remaining}`;
}

function phaseLabel(phase) {
  const labels = {
    queued: "Queued",
    starting: "Starting",
    downloading: "Downloading",
    extracting_audio: "Extracting audio",
    transcribing: "Transcribing",
    completed: "Completed",
    failed: "Failed"
  };
  return labels[phase] || phase || "Queued";
}

function statusTone(status) {
  if (status === "completed") return "success";
  if (status === "failed") return "danger";
  if (status === "running") return "active";
  return "muted";
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function getUrls(input) {
  return input
    .split(/\n|,/)
    .map((url) => url.trim())
    .filter(Boolean);
}

function MetricCard({ icon: Icon, label, value }) {
  return (
    <div className="metric-card">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function VideoMetaCard({ item }) {
  return (
    <div className="video-meta-card">
      <div className="video-meta-icon">
        <Video size={18} />
      </div>
      <div>
        <strong>{item.title || item.url}</strong>
        <span>
          {formatDuration(item.duration)}
          {item.uploader ? ` · ${item.uploader}` : ""}
          {item.ext ? ` · ${item.ext}` : ""}
        </span>
      </div>
    </div>
  );
}

function JobCard({ job, onCopy, onDownload, onRemove }) {
  const progress = Number(job.progress || 0);
  const firstVideo = job.videos?.[0];
  const firstMeta = job.videos?.find((video) => video.metadata)?.metadata;
  const title = firstMeta?.title || job.current_video || firstVideo?.url || "Transcription job";
  const tone = statusTone(job.status);

  return (
    <article className="job-card">
      <div className="job-card-header">
        <div>
          <span className={`status-pill ${tone}`}>{job.status || "queued"}</span>
          <h3>{title}</h3>
          <p>
            {job.error ||
              job.poll_warning ||
              `${phaseLabel(job.phase)} · ${job.completed_videos || 0}/${job.total_videos || 0} videos`}
          </p>
        </div>
        <div className="job-percent">{progress}%</div>
      </div>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="video-stack">
        {(job.videos || []).map((video, index) => (
          <div className="video-row" key={`${video.url}-${index}`}>
            <div>
              <strong>{video.metadata?.title || video.url}</strong>
              <span>{video.metadata?.duration ? formatDuration(video.metadata.duration) : phaseLabel(video.phase)}</span>
            </div>
            <b>{Number(video.progress || 0)}%</b>
          </div>
        ))}
      </div>

      {job.result?.transcript ? (
        <div className="transcript-panel">
          <div className="transcript-toolbar">
            <span>
              <FileText size={16} />
              Transcript
            </span>
            <div>
              <button type="button" className="icon-button" onClick={() => onCopy(job.result.transcript)} title="Copy transcript">
                <Clipboard size={16} />
              </button>
              <button type="button" className="icon-button" onClick={() => onDownload(job)} title="Download transcript">
                <Download size={16} />
              </button>
            </div>
          </div>
          <pre>{job.result.transcript}</pre>
        </div>
      ) : null}

      {job.status === "failed" || job.status === "completed" ? (
        <button type="button" className="remove-job" onClick={() => onRemove(job.job_id)}>
          <Trash2 size={15} />
          Remove
        </button>
      ) : null}
    </article>
  );
}

function App() {
  const [health, setHealth] = useState("Checking");
  const [urlInput, setUrlInput] = useState("");
  const [language, setLanguage] = useState("");
  const [metadata, setMetadata] = useState([]);
  const [metadataLoading, setMetadataLoading] = useState(false);
  const [jobs, setJobs] = useState(loadJobs);
  const [submitting, setSubmitting] = useState(false);
  const pollers = useRef(new Map());

  const urls = useMemo(() => getUrls(urlInput), [urlInput]);
  const running = jobs.filter((job) => job.status === "running" || job.status === "queued").length;
  const completed = jobs.filter((job) => job.status === "completed").length;

  useEffect(() => {
    saveJobs(jobs);
  }, [jobs]);

  useEffect(() => {
    requestJson("/health")
      .then(() => setHealth("Healthy"))
      .catch(() => setHealth("Offline"));
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(async () => {
      if (urls.length === 0) {
        setMetadata([]);
        return;
      }

      setMetadataLoading(true);
      try {
        setMetadata(await requestJson("/videos/metadata", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls })
        }));
      } catch {
        setMetadata([]);
      } finally {
        setMetadataLoading(false);
      }
    }, 600);

    return () => window.clearTimeout(timer);
  }, [urls.join("|")]);

  function upsertJob(update) {
    setJobs((current) => {
      const index = current.findIndex((job) => job.job_id === update.job_id);
      if (index === -1) return [update, ...current];
      const next = [...current];
      next[index] = { ...next[index], ...update };
      return next;
    });
  }

  function stopPolling(jobId) {
    const poller = pollers.current.get(jobId);
    if (poller) window.clearInterval(poller);
    pollers.current.delete(jobId);
  }

  function startPolling(jobId) {
    if (pollers.current.has(jobId)) return;

    const tick = async () => {
      try {
        const status = await requestJson(`/transcribe/jobs/${jobId}`);
        upsertJob({ ...status, poll_errors: 0, poll_warning: "" });

        if (status.status === "completed") {
          const result = await requestJson(`/transcribe/jobs/${jobId}/result`).catch(() => null);
          if (result) upsertJob({ ...status, result });
          stopPolling(jobId);
        }

        if (status.status === "failed") stopPolling(jobId);
      } catch (error) {
        const stored = loadJobs().find((job) => job.job_id === jobId) || { job_id: jobId, status: "queued" };
        const pollErrors = Number(stored.poll_errors || 0) + 1;
        if (pollErrors >= MAX_POLL_ERRORS) {
          upsertJob({ ...stored, status: "failed", poll_errors: pollErrors, error: error.message });
          stopPolling(jobId);
          return;
        }
        upsertJob({
          ...stored,
          status: "running",
          poll_errors: pollErrors,
          poll_warning: `Connection interrupted. Retrying ${pollErrors}/${MAX_POLL_ERRORS}...`
        });
      }
    };

    tick();
    pollers.current.set(jobId, window.setInterval(tick, 1800));
  }

  useEffect(() => {
    jobs
      .filter((job) => job.status === "queued" || job.status === "running")
      .forEach((job) => startPolling(job.job_id));
  }, []);

  async function submitJob(event) {
    event.preventDefault();
    if (urls.length === 0 || submitting) return;

    setSubmitting(true);
    try {
      const job = await requestJson("/transcribe/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls, language: language || undefined })
      });
      upsertJob({
        job_id: job.job_id,
        status: "queued",
        progress: 0,
        phase: "queued",
        total_videos: urls.length,
        completed_videos: 0,
        videos: urls.map((url, index) => ({
          url,
          metadata: metadata[index],
          status: "queued",
          progress: 0,
          phase: "queued"
        }))
      });
      setUrlInput("");
      setMetadata([]);
      startPolling(job.job_id);
    } catch (error) {
      alert(error.message);
    } finally {
      setSubmitting(false);
    }
  }

  function clearFinished() {
    setJobs((current) => current.filter((job) => job.status === "running" || job.status === "queued"));
  }

  function removeJob(jobId) {
    setJobs((current) => current.filter((job) => job.job_id !== jobId));
  }

  async function copyTranscript(text) {
    await navigator.clipboard.writeText(text);
  }

  function downloadTranscript(job) {
    const blob = new Blob([job.result.transcript], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `transcript-${job.job_id}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">HL</div>
          <div>
            <strong>Hesam Lecture</strong>
            <span>Transcriber</span>
          </div>
        </div>

        <nav>
          <a className="active"><Gauge size={18} /> Dashboard</a>
          <a><Activity size={18} /> Queue</a>
          <a><FileText size={18} /> Transcripts</a>
          <a><Server size={18} /> Local engine</a>
        </nav>

        <div className="engine-card">
          <HardDrive size={18} />
          <div>
            <span>Engine</span>
            <strong>Local Whisper</strong>
          </div>
        </div>
      </aside>

      <section className="main-panel">
        <header className="page-header">
          <div>
            <p className="eyebrow">Built by Hesam Hadadi</p>
            <h1>Lecture transcription dashboard</h1>
            <span>Manage video batches, monitor progress, and export cleaned transcripts.</span>
          </div>
          <div className={`health ${health.toLowerCase()}`}>
            <CheckCircle2 size={17} />
            {health}
          </div>
        </header>

        <section className="metrics-grid">
          <MetricCard icon={Video} label="Jobs" value={jobs.length} />
          <MetricCard icon={Loader2} label="Running" value={running} />
          <MetricCard icon={CheckCircle2} label="Completed" value={completed} />
        </section>

        <section className="workspace-grid">
          <form className="command-panel" onSubmit={submitJob}>
            <div className="panel-title">
              <div>
                <p className="eyebrow">New batch</p>
                <h2>Add lecture videos</h2>
              </div>
              <button type="button" className="icon-button" onClick={() => setUrlInput("")} title="Clear URLs">
                <RotateCcw size={17} />
              </button>
            </div>

            <label>
              Video URLs
              <textarea
                value={urlInput}
                onChange={(event) => setUrlInput(event.target.value)}
                placeholder="One URL per line"
              />
            </label>

            <div className="form-row">
              <label>
                Language
                <input value={language} onChange={(event) => setLanguage(event.target.value)} placeholder="auto, en, fa" />
              </label>
              <button type="submit" className="primary-button" disabled={submitting || urls.length === 0}>
                <Play size={18} />
                {submitting ? "Starting..." : "Start job"}
              </button>
            </div>
          </form>

          <aside className="preview-panel">
            <div className="panel-title">
              <div>
                <p className="eyebrow">Preview</p>
                <h2>Video details</h2>
              </div>
              <Info size={18} />
            </div>
            {metadataLoading ? (
              <div className="empty-state"><Loader2 size={18} /> Loading metadata...</div>
            ) : metadata.length > 0 ? (
              <div className="metadata-list">{metadata.map((item) => <VideoMetaCard key={item.url} item={item} />)}</div>
            ) : (
              <div className="empty-state"><Link size={18} /> Paste URLs to inspect videos.</div>
            )}
          </aside>
        </section>

        <section className="queue-panel">
          <div className="panel-title">
            <div>
              <p className="eyebrow">Live queue</p>
              <h2>Task management</h2>
            </div>
            <button type="button" className="secondary-button" onClick={clearFinished}>
              <Trash2 size={16} />
              Clear finished
            </button>
          </div>

          {jobs.length > 0 ? (
            <div className="jobs-list">
              {jobs.map((job) => (
                <JobCard
                  key={job.job_id}
                  job={job}
                  onCopy={copyTranscript}
                  onDownload={downloadTranscript}
                  onRemove={removeJob}
                />
              ))}
            </div>
          ) : (
            <div className="empty-state large">
              <Sparkles size={24} />
              No jobs yet. Add videos above to start a managed transcription batch.
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
