"""
Custom tools for the Groq GAIA agent.

Design notes
------------
* The reasoning LLM (agent.py) runs on Groq via LiteLLM.
* The multimodal tools also run on Groq:
    - analyze_image  -> Groq vision model (default qwen/qwen3.6-27b)
    - transcribe_audio -> Groq Whisper (default whisper-large-v3-turbo)
  So a single GROQ_API_KEY covers text, image, and audio. No Gemini/OpenAI key
  needed.
* Every tool returns a *string*. Tools never raise into the agent loop on an
  expected failure; they return an "ERROR: ..." string so the model can react
  (retry a different tool, rephrase) instead of the whole run crashing.
* Tool outputs are truncated. Groq's per-minute token cap is tight and the
  CodeAgent keeps every past observation in context, so small outputs matter.
"""

import os
import json
import base64
import mimetypes

from smolagents import tool

# Model ids (overridable) — kept in sync with agent.py's CONFIG philosophy.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "groq/qwen/qwen3.6-27b")
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# Truncation guard. Keep tool outputs well under Groq's TPM budget.
_MAX_CHARS = 2500


def _truncate(text: str, limit: int = _MAX_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n...[truncated, {len(text) - limit} more chars]"


# --------------------------------------------------------------------------- #
# File reading                                                                #
# --------------------------------------------------------------------------- #
@tool
def read_file(file_path: str) -> str:
    """Read a local file and return its content as text.

    Dispatches by extension: .txt/.md/.py/.json as text, .csv/.xlsx/.xls as a
    tabular dump via pandas, .pdf via pdfplumber. Use this to inspect any file
    that was attached to a question. For images use analyze_image; for audio use
    transcribe_audio.

    Args:
        file_path: Absolute path to the local file to read.
    """
    if not os.path.exists(file_path):
        return f"ERROR: file not found at {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in {".txt", ".md", ".py", ".log", ".xml", ".html", ".tsv"}:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return _truncate(f.read())

        if ext == ".json":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return _truncate(json.dumps(json.load(f), indent=2, ensure_ascii=False))

        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            return _truncate(f"CSV shape {df.shape}\n\n{df.to_string()}")

        if ext in {".xlsx", ".xls"}:
            import pandas as pd
            sheets = pd.read_excel(file_path, sheet_name=None)
            out = [f"=== sheet: {name} (shape {df.shape}) ===\n{df.to_string()}"
                   for name, df in sheets.items()]
            return _truncate("\n\n".join(out))

        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return _truncate("\n\n".join(pages))

        # Unknown extension: best-effort text read.
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return _truncate(f.read())

    except Exception as e:
        return f"ERROR reading {file_path}: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Vision (Groq multimodal)                                                    #
# --------------------------------------------------------------------------- #
def _data_uri(file_path: str) -> str:
    """Encode a local image as a base64 data URI for the OpenAI/Groq image API."""
    mime = mimetypes.guess_type(file_path)[0] or "image/png"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@tool
def analyze_image(file_path: str, question: str) -> str:
    """Answer a question about an image (photo, chart, diagram, screenshot,
    chessboard). Handles OCR (reading text in the image) too. Use for any
    question whose attached file is an image.

    Args:
        file_path: Absolute path to the local image file.
        question: The specific question to answer about the image.
    """
    if not os.path.exists(file_path):
        return f"ERROR: image not found at {file_path}"
    try:
        import litellm
        resp = litellm.completion(
            model=VISION_MODEL,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url",
                     "image_url": {"url": _data_uri(file_path)}},
                ],
            }],
        )
        return _truncate(resp.choices[0].message.content or "")
    except Exception as e:
        # Fallback: local OCR if the vision model is unavailable / rate-limited.
        try:
            from PIL import Image
            from pytesseract import image_to_string
            text = image_to_string(Image.open(file_path))
            if text.strip():
                return _truncate("Vision model unavailable; OCR text only:\n" + text)
        except Exception:
            pass
        return f"ERROR analyzing image: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Audio (Groq Whisper)                                                        #
# --------------------------------------------------------------------------- #
@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe a local audio file (mp3, wav, m4a, flac) to text using Groq
    Whisper. Use for any question whose attached file is audio, e.g. a voice
    memo whose contents you must read.

    Args:
        file_path: Absolute path to the local audio file.
    """
    if not os.path.exists(file_path):
        return f"ERROR: audio not found at {file_path}"
    try:
        import litellm
        with open(file_path, "rb") as f:
            resp = litellm.transcription(
                model=f"groq/{WHISPER_MODEL}",
                file=f,
                temperature=0.0,
                response_format="text",
            )
        # litellm returns an object with .text (or a raw string for some versions)
        text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else str(resp))
        return _truncate(text)
    except Exception as e:
        return f"ERROR transcribing audio: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# YouTube                                                                     #
# --------------------------------------------------------------------------- #
@tool
def get_youtube_transcript(url_or_id: str) -> str:
    """Fetch the transcript of a YouTube video. Use when a question references a
    YouTube video and the answer is in what is said. Accepts a full URL or a
    bare 11-character video id.

    Args:
        url_or_id: A YouTube URL or the raw video id.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception as e:
        return f"ERROR: youtube-transcript-api not installed: {e}"

    vid = url_or_id.strip()
    if "watch?v=" in vid:
        vid = vid.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in vid:
        vid = vid.split("youtu.be/")[1].split("?")[0]
    elif "shorts/" in vid:
        vid = vid.split("shorts/")[1].split("?")[0]

    try:
        fetched = YouTubeTranscriptApi().fetch(vid)  # >=1.0 instance API
        try:
            segments = fetched.to_raw_data()
            text = " ".join(s["text"] for s in segments)
        except AttributeError:
            text = " ".join(sn.text for sn in fetched)
        return _truncate(text)
    except Exception as e:
        return f"ERROR fetching transcript for '{vid}': {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Web page reader                                                             #
# --------------------------------------------------------------------------- #
def _fetch_page_text(url: str) -> str:
    """Fetch a URL and return cleaned main text (may be long)."""
    import re
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GaiaAgent/1.0)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    text = None
    try:
        import trafilatura
        text = trafilatura.extract(r.text, include_links=False, include_tables=True)
    except Exception:
        text = None
    if not text:
        try:
            from markdownify import markdownify as md
            text = md(r.text)
        except Exception:
            text = r.text
    return re.sub(r"\n{3,}", "\n\n", text or "").strip()


@tool
def visit_webpage(url: str, search: str = "") -> str:
    """Fetch a web page and return its MAIN text content (navigation/boilerplate
    stripped, tables kept). Use to read a specific URL found via web_search.

    IMPORTANT: the page is long and only a window of it is returned. If the part
    you need (e.g. a "Discography" or "Studio albums" table) is not in the
    default window, call this again with the `search` argument set to a keyword
    that appears near it — the returned window will be centered on the first
    match instead of the top of the page. This is how you reach content deep in
    a long article. Do NOT re-fetch with the same arguments expecting a
    different result.

    Args:
        url: The full URL to fetch.
        search: Optional keyword. If given, return the window of text around the
            first occurrence of this keyword (case-insensitive) instead of the
            page top. Example: search="Discography".
    """
    try:
        text = _fetch_page_text(url)
    except Exception as e:
        return f"ERROR fetching {url}: {type(e).__name__}: {e}"

    if not text:
        return f"ERROR: no readable text extracted from {url}"

    if search:
        idx = text.lower().find(search.lower())
        if idx == -1:
            # Report what headings DO exist so the agent can pick a real one.
            import re
            heads = re.findall(r"^#{1,6}\s*(.+)$|^\s*(.+)\n[=-]{3,}\s*$",
                               text, flags=re.MULTILINE)
            flat = [h[0] or h[1] for h in heads][:30]
            hint = ("; ".join(flat) if flat
                    else "no clear headings found")
            return (f"'{search}' not found on page. Sections/headings present: "
                    f"{hint}. Total page length {len(text)} chars.")
        start = max(0, idx - 200)
        window = text[start:start + _MAX_CHARS]
        return (f"[window around '{search}', page is {len(text)} chars total]\n"
                + window)

    # No search term: return the top window, but say how much more there is.
    if len(text) > _MAX_CHARS:
        return (f"[showing first {_MAX_CHARS} of {len(text)} chars; call again "
                f"with search='<keyword>' to jump to a later section]\n"
                + text[:_MAX_CHARS])
    return text
