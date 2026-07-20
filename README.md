# gaia-groq
---
title: GAIA Agent (Groq)
emoji: 👽
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Benchmark Agent — Groq edition

A [smolagents](https://github.com/huggingface/smolagents) `CodeAgent` for the
Hugging Face Agents Course final project (the [GAIA](https://huggingface.co/learn/agents-course/en/unit4/what-is-gaia)
subset), running **entirely on [Groq](https://groq.com)**.

Text reasoning, image understanding, and audio transcription all go through a
single provider and a single `GROQ_API_KEY` — Groq hosts a vision model *and*
Whisper, so there is no second API key to manage.

## Models

Groq rotates its line-up often. These are the defaults (July 2026); each is
overridable by an environment variable so you never have to edit code when Groq
deprecates one.

| Role      | Default id                    | Env override          |
| --------- | ----------------------------- | --------------------- |
| Reasoning | `groq/openai/gpt-oss-120b`    | `GROQ_MODEL_ID`       |
| Vision    | `groq/qwen/qwen3.6-27b`       | `GROQ_VISION_MODEL`   |
| Audio     | `whisper-large-v3-turbo`      | `GROQ_WHISPER_MODEL`  |

> Note: on 2026-06-17 Groq deprecated `llama-3.3-70b-versatile`,
> `llama-3.1-8b-instant`, and `llama-4-scout`. The defaults above are the
> current recommended replacements. Check the
> [Groq model list](https://console.groq.com/docs/models) if a call 404s.

## Tools

| Tool | Backend | Handles |
| ---- | ------- | ------- |
| `DuckDuckGoSearchTool` | web | open web search |
| `WikipediaSearchTool` (summary) | Wikipedia | encyclopedic lookups, small outputs |
| `visit_webpage` | requests + trafilatura | read one URL, tables kept |
| `read_file` | pandas / pdfplumber | txt, md, json, csv, xlsx, pdf |
| `analyze_image` | **Groq vision** | photos, charts, screenshots, OCR (OCR fallback if vision is down) |
| `transcribe_audio` | **Groq Whisper** | mp3, wav, m4a, flac |
| `get_youtube_transcript` | youtube-transcript-api | spoken content of a video |

## Design choices for the GAIA benchmark

- **Rate-limit resilience.** Groq's free/dev tiers cap tokens-per-minute, and a
  `CodeAgent` carries its whole trajectory in context. `GroqLiteLLMModel` catches
  Groq's `RateLimitError`, parses the suggested wait ("try again in 4.2s") and
  retries with exponential backoff — instead of losing the question.
- **Answer normalization.** GAIA is exact-match graded. `normalize_answer`
  strips `FINAL ANSWER:` prefixes, quotes and trailing periods, and reduces bare
  numbers to `1234` (no separators, `$`, or `%`).
- **Reversed-text handling.** GAIA includes a question written backwards;
  `looks_reversed` detects it and hands the model both orientations.
- **Small tool outputs.** Every tool truncates to ~2.5k chars and Wikipedia runs
  in summary mode, to stay inside Groq's per-minute token budget.

## Setup

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env      # get one at console.groq.com
```

## Run

**Locally / in Colab** (no HF Space needed):

```bash
export GROQ_API_KEY=...
python run_local.py --refresh            # run all cached questions
python run_local.py --id 3 --trace       # one question, full reasoning trace
python run_local.py --submit --username <you> \
    --agent-code https://github.com/<you>/gaia-agent/tree/main
```

**As a Gradio app / HF Space:**

```bash
python app.py
```

The `GaiaAgent(question, file_path)` interface is unchanged, so the existing
`app.py` and `run_local.py` work as-is.
