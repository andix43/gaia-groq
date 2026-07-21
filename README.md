---
title: GAIA Agent (Groq)
emoji: 🏆
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Agent — Groq edition

A [smolagents](https://github.com/huggingface/smolagents) `CodeAgent` for the
Hugging Face Agents Course final project (the 20-question Level-1 subset of
[GAIA](https://huggingface.co/learn/agents-course/en/unit4/what-is-gaia)),
running **entirely on [Groq](https://groq.com)**.

Text reasoning, image understanding, and audio transcription all go through a
single provider and a single `GROQ_API_KEY` — Groq hosts a multimodal Qwen model
*and* Whisper, so there is no second API key to manage.

The design priority is **reliable completion**: the agent is built to run through
all 20 questions without hanging or crashing, using bounded time and steps per
question, rather than to squeeze out maximum accuracy.

---

## Models

| Role      | Default id                   | Env override          |
| --------- | ---------------------------- | --------------------- |
| Reasoning | `groq/qwen/qwen3.6-27b`      | `GROQ_MODEL_ID`       |
| Vision    | `groq/qwen/qwen3.6-27b`      | `GROQ_VISION_MODEL`   |
| Audio     | `whisper-large-v3-turbo`     | `GROQ_WHISPER_MODEL`  |

Qwen is used deliberately: Groq's `gpt-oss` models emit native tool calls under
`CodeAgent` (which runs with `tool_choice=none`), tripping Groq's
`tool_use_failed` error and mangling code blobs. Qwen writes smolagents-style
code blocks reliably and is multimodal, so it also powers image analysis.

Groq rotates model ids often. If a call 404s, check the
[Groq model list](https://console.groq.com/docs/models) and set `GROQ_MODEL_ID`.

---

## Tools

| Tool | Backend | Handles |
| ---- | ------- | ------- |
| `DuckDuckGoSearchTool` | web | open web search |
| `WikipediaSearchTool` (summary) | Wikipedia | encyclopedic lookups |
| `visit_webpage(url, search=…)` | requests + trafilatura | read one URL; `search=` jumps to a section on long pages |
| `read_file` | pandas / pdfplumber / python-docx / python-pptx | txt, md, py, json, csv, xlsx, pdf, docx, pptx — local path **or** http URL |
| `analyze_image` | **Groq vision** | photos, charts, chessboards, screenshots, OCR (local OCR fallback) |
| `transcribe_audio` | **Groq Whisper** | mp3, wav, m4a, flac |
| `get_youtube_transcript` | youtube-transcript-api | spoken content of a video |
| `analyze_youtube_video` | yt-dlp + Groq vision | visual content of a video (frame sampling) |

The CodeAgent can also run Python directly (pandas, numpy, math, itertools,
base64/codecs for decoding, etc.) for pure reasoning and data questions.

---

## Reliability features

These exist to survive Groq's free tier and GAIA's harder edge cases:

- **Per-question sleep budget** (`GAIA_MAX_QUESTION_SLEEP`, default 150s). When
  rate-limit sleeps for a single question exceed the budget, the agent gives up
  on that question gracefully instead of blocking the whole run.
- **Hard per-question timeout** in `run_local.py` (`--timeout`, default 300s). A
  signal alarm interrupts anything — a sleep, a hung download, a tool loop — and
  records `ERROR` so the run always advances.
- **Bounded steps** (`GAIA_MAX_STEPS`, default 4) plus a "if a tool fails twice,
  answer anyway" hint, so a question can't burn many rate-limited LLM calls.
- **`tool_use_failed` recovery** — if a model emits a rejected native tool call,
  the intended code is recovered from the error and re-run as a code blob.
- **`<think>` stripping** — Qwen reasoning blocks are removed from stored
  messages and final answers (keeps context small; stops reasoning leaking into
  an answer).
- **Answer normalization** toward GAIA's exact-match format (strips labels,
  quotes, stray code tags; renders bare numbers as `1234`).
- **Reversed-text handling** for GAIA's backwards-written question.
- **Per-question error isolation** in `run_local.py` — one bad question (missing
  file, exception, interrupt) never aborts the run.

---

## Setup

```bash
pip install -r requirements.txt
# optional: enables the OCR fallback and video frames
sudo apt-get install -y tesseract-ocr ffmpeg
export GROQ_API_KEY=...            # from console.groq.com
```

## Run

**Locally / in Colab** (no HF Space needed):

```bash
python run_local.py --refresh --id 5 --trace     # one question, full trace
python run_local.py --refresh --timeout 240      # all 20, each capped at 240s
python run_local.py --submit \
    --username <you> \
    --agent-code https://huggingface.co/spaces/<you>/<space>/tree/main
```

A ready-made Colab notebook (`gaia_groq_runner.ipynb`) is included.

**As a Gradio app / HF Space:** `python app.py` — log in with Hugging Face and
click **Run & Submit All**. Keep the Space public so `agent_code` is verifiable.

Note: the scoring API only records the `agent_code` URL; it does not execute your
Space. You can run the eval in Colab and submit with your Space URL as
`--agent-code`.

## Configuration (env vars)

| Var | Default | Purpose |
| --- | ------- | ------- |
| `GROQ_API_KEY` | — | required |
| `GROQ_MODEL_ID` | `groq/qwen/qwen3.6-27b` | reasoning + vision model |
| `GROQ_VISION_MODEL` | = `GROQ_MODEL_ID` | image model |
| `GROQ_WHISPER_MODEL` | `whisper-large-v3-turbo` | audio model |
| `GAIA_MAX_STEPS` | `4` | max agent steps per question |
| `GAIA_MAX_QUESTION_SLEEP` | `150` | max rate-limit sleep (s) per question |
| `YOUTUBE_PROXY` | — | residential proxy for YouTube (cloud IPs are blocked) |

---

## Known limitations

This is a course project on a free tier, and some failures are structural, not
bugs:

- **Free-tier rate limits are the dominant cost.** Groq's per-minute token cap
  forces long sleeps between steps; on a full run several winnable questions time
  out or return an empty generation after a long wait. A **Dev-tier Groq key**
  removes the cap and is the single biggest improvement available, with no code
  change.
- **YouTube is IP-blocked from Colab / cloud IPs.** Both `yt-dlp` and the
  transcript API are blocked from Google-Cloud ranges. Video questions need a
  residential `YOUTUBE_PROXY`, or a home connection. Transcript questions can
  still work when subtitles are available.
- **Attached-file questions depend on the scoring API serving the file.** When
  `GET /files/{task_id}` returns 404, the agent has nothing to read.
- **Search results can be adversarial.** GAIA-related queries sometimes surface
  SEO/injection pages; the agent does not blindly trust them, but this is a known
  hazard of open web search.

Scoring is **exact string match**, so answer formatting matters as much as being
right; the normalization step targets this, but skim the printed `A: '...'`
lines before submitting. The course bar is **30% on Level 1** (~6/20).

---

## Files

| File | Purpose |
| ---- | ------- |
| `agent.py` | `GaiaAgent`, the Groq-hardened model wrapper, normalization |
| `tools.py` | custom tools (file/image/audio/youtube/webpage) |
| `run_local.py` | CLI runner with per-question timeout + submission |
| `app.py` | Gradio app for the submission Space |
| `gaia_groq_runner.ipynb` | Colab notebook |
| `requirements.txt` | dependencies |
