---
title: GAIA Agent (Groq / Gemini)
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

# GAIA Agent

A [smolagents](https://github.com/huggingface/smolagents) `CodeAgent` for the
Hugging Face Agents Course final project — the 20-question Level-1 subset of
[GAIA](https://huggingface.co/learn/agents-course/en/unit4/what-is-gaia).

Provider-flexible: runs on **Google Gemini** or **Groq** from a single code
base, selected by whichever API key is present. One key covers text reasoning,
vision, and (on Groq) audio.

The design priority is **reliable completion on a free tier**. The agent bounds
its own time, steps, and API-request usage, skips questions that are provably
unanswerable, and recovers answers it computed but failed to submit. A recent
full run finished all 20 questions in **8m35s using 109 API requests**.

---

## Models

| Role      | Default                        | Override             |
| --------- | ------------------------------ | -------------------- |
| Reasoning | `gemini/gemini-3.1-flash-lite` | `AGENT_MODEL_ID`     |
| Vision    | same as reasoning              | `AGENT_VISION_MODEL` |
| Audio     | `whisper-large-v3-turbo` (Groq)| `GROQ_WHISPER_MODEL` |

Selection order: `AGENT_MODEL_ID` → `GEMINI_API_KEY` → `GROQ_API_KEY`.

**Why Flash-Lite by default.** On Gemini's free tier the binding limit is
requests-per-day, and it differs enormously by model:

| Model | RPM | TPM | RPD |
| ----- | --- | --- | --- |
| `gemini-3.1-flash-lite` | 15 | 250K | **500** |
| `gemini-3.6-flash`      |  5 | 250K | 20 |
| `gemini-3-flash`        |  5 | 250K | 20 |

A full run costs ~110 requests, so Flash-Lite is the only free model that can
finish one. With billing enabled, `gemini/gemini-3.6-flash` is the stronger
model. Google retires model ids frequently, so the agent carries a fallback
chain and switches automatically on a deprecation 404.

Avoid Groq's `gpt-oss` models: under `CodeAgent` (which runs with
`tool_choice=none`) they emit native tool calls, tripping `tool_use_failed`, and
they mangle code blobs. `groq/qwen/qwen3.6-27b` is the Groq model to use.

---

## Tools

| Tool | Backend | Handles |
| ---- | ------- | ------- |
| `DuckDuckGoSearchTool` | web | open web search |
| `WikipediaSearchTool` (summary) | Wikipedia | encyclopedic lookups |
| `visit_webpage(url, search=…)` | requests + trafilatura | read a URL; `search=` jumps to a section instead of re-fetching a long page |
| `read_file` | pandas / pdfplumber / python-docx / python-pptx | txt, md, py, json, csv, xlsx, pdf, docx, pptx — local path **or** http URL |
| `analyze_image` | vision model | charts, chessboards, screenshots, OCR (local OCR fallback) |
| `transcribe_audio` | Groq Whisper | mp3, wav, m4a, flac |
| `get_youtube_transcript` | youtube-transcript-api | spoken content of a video |

The agent also writes and runs Python directly (pandas, numpy, itertools,
base64/codecs, …) for computation, decoding, and table analysis.

`analyze_youtube_video` (frame sampling + vision) exists in `tools.py` but is
not wired into the agent: YouTube blocks video downloads from cloud IPs, so it
only ever cost a step. Enable it in `build_agent()` if running behind a
residential connection or `YOUTUBE_PROXY`.

---

## Reliability

Everything here came from a failure observed in a real run.

**Quota safety**
- *Self-throttling* — enforces `GAIA_MIN_REQUEST_INTERVAL` (4.2s) between calls,
  staying under the per-minute cap so 429s never occur.
- *Request budget* — stops cleanly at `GAIA_MAX_REQUESTS` (400) with answers
  intact and submittable, instead of being cut off by the daily cap.
- *Skips unanswerable questions* — if the scoring API serves no file for a task,
  it is skipped with zero model calls (see Known limitations).

**Not losing a computed answer**
- *Empty-generation handling* — Gemini 3.x often returns only "thought" tokens
  with an empty text field. The agent disables thinking, digs the text out of
  `reasoning_content` / the raw provider payload, and retries adaptively
  (dropping `stop_sequences` on retry) rather than repeating a failing call.
- *Answer salvage* — if the agent prints the answer but dies before calling
  `final_answer`, the printed value is recovered. Deliberately conservative:
  error text, search dumps, and page windows are never accepted.
- *`tool_use_failed` recovery* — recovers the rejected code from the error
  payload and runs it.
- *`<think>` stripping* — removes reasoning blocks, but never empties a message
  (doing so was itself a bug that cost ~12 questions in one run).

**Bounded work**
- Hard per-question wall-clock timeout (`--timeout`, default 300s) via signal
  alarm — interrupts sleeps, hung downloads, and tool loops alike.
- `GAIA_MAX_STEPS` (6) and a per-question rate-limit sleep budget.
- Per-question error isolation: one bad question never aborts the run.

**Scoring**
- Answer normalization for GAIA's exact-match grading (strips labels, quotes,
  stray code tags; bare numbers rendered as `1234`).
- Reversed-text detection for GAIA's backwards-written question.

---

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install -y tesseract-ocr ffmpeg     # optional: OCR fallback
export GEMINI_API_KEY=...        # from aistudio.google.com  (recommended)
# or: export GROQ_API_KEY=...    # from console.groq.com
```

## Run

```bash
python run_local.py --refresh --id 8 --trace   # one question, full trace
python run_local.py --refresh --timeout 300    # all questions
```

The runner prints a summary of which questions produced answers and how many
API requests were used.

## Submit

The scoring API only records the `agent_code` URL — it does not execute your
Space. So you can run the eval locally and submit with your public Space URL.

Submit the answers from a run you already inspected (no re-running, no extra
quota, no answer drift):

```bash
# edit the ANSWERS dict in submit_answers.py, then:
python submit_answers.py --username <you> \
    --agent-code https://huggingface.co/spaces/<you>/<space>/tree/main --dry-run
```

Or run and submit in one shot:

```bash
python run_local.py --submit --username <you> \
    --agent-code https://huggingface.co/spaces/<you>/<space>/tree/main
```

Or use the Gradio app (`python app.py`) — log in with Hugging Face and click
**Run & Submit All**. Keep the Space public.

The course bar is **30%** (6/20 correct). Submissions can be repeated.

## Configuration

| Var | Default | Purpose |
| --- | ------- | ------- |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | — | one is required |
| `AGENT_MODEL_ID` | auto | explicit LiteLLM model id |
| `AGENT_VISION_MODEL` | = reasoning model | image model |
| `GROQ_WHISPER_MODEL` | `whisper-large-v3-turbo` | audio model |
| `GAIA_MAX_STEPS` | `6` | max agent steps per question |
| `GAIA_MAX_QUESTION_SLEEP` | `180` | max rate-limit sleep (s) per question |
| `GAIA_MIN_REQUEST_INTERVAL` | `4.2` | seconds between API calls (RPM guard) |
| `GAIA_MAX_REQUESTS` | `400` | stop the run before the daily cap |
| `YOUTUBE_PROXY` | — | residential proxy for YouTube |

---

## Known limitations

Some failures here are structural, not bugs:

- **The five attachment questions are unanswerable.** `GET /files/{task_id}`
  returns `{"detail":"No file path associated with task_id ..."}` for every task
  that `/questions` says has a `file_name`. Verified from both a residential and
  a cloud IP, so it is a server-side inconsistency in the course API, not a
  client problem. The agent detects this and skips them without spending calls.
  Effective denominator is 15, but the leaderboard still scores out of 20.
- **YouTube is blocked from cloud IPs.** Video download and, for some videos,
  transcript fetching fail from Colab. Transcript questions succeed when
  subtitles exist (that is how question 6 is answered).
- **Free-tier rate limits shape the design.** Groq's free tier caps at 6K TPM,
  which made multi-step questions stall; Gemini's free tier caps requests per
  day. The throttle and budget exist because of this.
- **Search results can be adversarial.** GAIA-related queries surface SEO and
  prompt-injection pages, some containing fabricated "answers". Treat retrieved
  text as untrusted.
- **Multi-hop web questions are the weak spot.** Chained lookups (find X, then
  find Y about X) are where a small Flash-Lite model most often stops short.

Realistic expectation: **6–9 answers out of 20**, of which not all will be
correct. GAIA is hard by design — the published human baseline is ~92% while
strong agents land in the 30–50% range.

---

## Files

| File | Purpose |
| ---- | ------- |
| `agent.py` | `GaiaAgent`, hardened model wrapper, salvage, normalization |
| `tools.py` | custom tools (file / image / audio / youtube / webpage) |
| `run_local.py` | CLI runner: timeouts, skipping, submission |
| `submit_answers.py` | submit a fixed answer set without re-running |
| `check_files.py` | diagnose whether the scoring API serves attachments |
| `app.py` | Gradio app for the submission Space |
| `requirements.txt` | dependencies |
