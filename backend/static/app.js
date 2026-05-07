const form = document.querySelector("#jobForm");
const urlsInput = document.querySelector("#urls");
const languageInput = document.querySelector("#language");
const jobsContainer = document.querySelector("#jobs");
const jobTemplate = document.querySelector("#jobTemplate");
const healthState = document.querySelector("#healthState");
const jobCount = document.querySelector("#jobCount");
const runningCount = document.querySelector("#runningCount");
const completedCount = document.querySelector("#completedCount");
const clearUrls = document.querySelector("#clearUrls");
const clearJobs = document.querySelector("#clearJobs");

const STORAGE_KEY = "lecture-transcriber-jobs";
const MAX_POLL_ERRORS = 6;
const pollers = new Map();
let jobs = loadJobs();

function loadJobs() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveJobs() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs));
}

function phaseLabel(phase) {
  const labels = {
    queued: "Queued",
    starting: "Starting",
    downloading: "Downloading",
    extracting_audio: "Extracting audio",
    transcribing: "Transcribing",
    completed: "Completed",
    failed: "Failed",
  };
  return labels[phase] || phase;
}

function statusLabel(status) {
  const labels = {
    queued: "Queued",
    running: "Running",
    completed: "Completed",
    failed: "Failed",
  };
  return labels[status] || status;
}

function getUrls() {
  return urlsInput.value
    .split(/\n|,/)
    .map((url) => url.trim())
    .filter(Boolean);
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Health check failed");
    healthState.textContent = "Healthy";
  } catch {
    healthState.textContent = "Offline";
  }
}

async function createJob(urls, language) {
  const payload = { urls };
  if (language) payload.language = language;

  const response = await fetch("/transcribe/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not create job");
  }

  return response.json();
}

async function fetchStatus(jobId) {
  const response = await fetch(`/transcribe/jobs/${jobId}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not fetch job status");
  }
  return response.json();
}

async function fetchResult(jobId) {
  const response = await fetch(`/transcribe/jobs/${jobId}/result`);
  if (response.status === 202) return null;
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not fetch transcript");
  }
  return response.json();
}

function upsertJob(update) {
  const index = jobs.findIndex((job) => job.job_id === update.job_id);
  if (index >= 0) {
    jobs[index] = { ...jobs[index], ...update };
  } else {
    jobs.unshift(update);
  }
  saveJobs();
  renderJobs();
}

function getStoredJob(jobId) {
  return jobs.find((job) => job.job_id === jobId);
}

function startPolling(jobId) {
  if (pollers.has(jobId)) return;

  const tick = async () => {
    try {
      const status = await fetchStatus(jobId);
      upsertJob({ ...status, poll_errors: 0, poll_warning: "" });

      if (status.status === "completed") {
        const result = await fetchResult(jobId);
        if (result) upsertJob({ ...status, poll_errors: 0, poll_warning: "", result });
        stopPolling(jobId);
      }

      if (status.status === "failed") stopPolling(jobId);
    } catch (error) {
      const current = getStoredJob(jobId) || { job_id: jobId, status: "queued" };
      const pollErrors = Number(current.poll_errors || 0) + 1;

      if (error.message === "Job not found" || pollErrors >= MAX_POLL_ERRORS) {
        upsertJob({
          ...current,
          status: "failed",
          poll_errors: pollErrors,
          error:
            error.message === "Job not found"
              ? "This job was lost, likely because the API restarted."
              : error.message,
        });
        stopPolling(jobId);
        return;
      }

      upsertJob({
        ...current,
        status: current.status === "completed" ? "completed" : "running",
        poll_errors: pollErrors,
        poll_warning: `Connection hiccup. Retrying ${pollErrors}/${MAX_POLL_ERRORS}...`,
      });
    }
  };

  tick();
  pollers.set(jobId, window.setInterval(tick, 1800));
}

function stopPolling(jobId) {
  const poller = pollers.get(jobId);
  if (poller) window.clearInterval(poller);
  pollers.delete(jobId);
}

function renderVideoRows(videos = []) {
  return videos
    .map(
      (video) => `
        <div class="video-row">
          <div>
            <div class="video-url">${escapeHtml(video.url)}</div>
            <div class="video-phase">${phaseLabel(video.phase)}</div>
          </div>
          <div class="video-percent">${Number(video.progress || 0)}%</div>
        </div>
      `,
    )
    .join("");
}

function renderJobs() {
  jobsContainer.innerHTML = "";
  jobsContainer.classList.toggle("empty", jobs.length === 0);

  if (jobs.length === 0) {
    jobsContainer.innerHTML = "<p>No jobs yet. Add a few lecture links and start a background job.</p>";
  }

  let running = 0;
  let completed = 0;

  jobs.forEach((job, index) => {
    if (job.status === "running" || job.status === "queued") running += 1;
    if (job.status === "completed") completed += 1;

    const node = jobTemplate.content.firstElementChild.cloneNode(true);
    const pill = node.querySelector(".pill");
    const title = node.querySelector("h3");
    const percent = node.querySelector(".job-percent");
    const fill = node.querySelector(".progress-fill");
    const meta = node.querySelector(".job-meta");
    const videoList = node.querySelector(".video-list");
    const transcript = node.querySelector(".transcript");
    const pre = node.querySelector("pre");
    const copyButton = node.querySelector(".copy-button");
    const downloadButton = node.querySelector(".download-button");

    const progress = Number(job.progress || 0);
    pill.textContent = statusLabel(job.status || "queued");
    pill.classList.toggle("completed", job.status === "completed");
    pill.classList.toggle("failed", job.status === "failed");
    title.textContent = job.current_video || job.videos?.[0]?.url || `Job ${index + 1}`;
    percent.textContent = `${progress}%`;
    fill.style.width = `${progress}%`;
    meta.textContent = `${phaseLabel(job.phase)} · ${job.completed_videos || 0}/${job.total_videos || 0} videos`;
    videoList.innerHTML = renderVideoRows(job.videos);

    if (job.poll_warning) {
      meta.textContent = job.poll_warning;
    }

    if (job.error) {
      meta.textContent = job.error;
    }

    if (job.result?.transcript) {
      transcript.classList.remove("hidden");
      pre.textContent = job.result.transcript;
      copyButton.addEventListener("click", () => navigator.clipboard.writeText(job.result.transcript));
      downloadButton.addEventListener("click", () => downloadTranscript(job));
    }

    jobsContainer.appendChild(node);
  });

  jobCount.textContent = jobs.length;
  runningCount.textContent = running;
  completedCount.textContent = completed;
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

function escapeHtml(value = "") {
  return value.replace(/[&<>"']/g, (char) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return entities[char];
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = form.querySelector(".primary-button");
  const urls = getUrls();
  const language = languageInput.value.trim();

  if (urls.length === 0) return;

  submit.disabled = true;
  submit.textContent = "Starting...";

  try {
    const job = await createJob(urls, language);
    upsertJob({
      job_id: job.job_id,
      status: "queued",
      progress: 0,
      phase: "queued",
      total_videos: urls.length,
      completed_videos: 0,
      videos: urls.map((url) => ({ url, status: "queued", progress: 0, phase: "queued" })),
    });
    startPolling(job.job_id);
    urlsInput.value = "";
  } catch (error) {
    alert(error.message);
  } finally {
    submit.disabled = false;
    submit.textContent = "Start Background Job";
  }
});

clearUrls.addEventListener("click", () => {
  urlsInput.value = "";
});

clearJobs.addEventListener("click", () => {
  jobs = jobs.filter((job) => job.status === "running" || job.status === "queued");
  saveJobs();
  renderJobs();
});

checkHealth();
renderJobs();
jobs
  .filter((job) => job.status === "queued" || job.status === "running")
  .forEach((job) => startPolling(job.job_id));
