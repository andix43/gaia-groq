"""
GaiaAgent — a smolagents CodeAgent tuned for the HF Agents Course GAIA subset,
running entirely on Groq.

Why Groq
--------
Groq serves open models on LPU hardware at very high throughput, and — crucially
for GAIA — it hosts BOTH a vision model and Whisper speech-to-text, so text,
image, and audio questions can all be answered on a single provider with one API
key. The reasoning LLM, the vision model, and the audio model are each swappable
via environment variables (see the CONFIG block below).

Model IDs (defaults, July 2026)
-------------------------------
Groq's line-up changes fast. On 2026-06-17 Groq deprecated
``llama-3.3-70b-versatile``, ``llama-3.1-8b-instant`` and
``llama-4-scout-17b-16e-instruct``. The current recommended replacements are:

    reasoning : openai/gpt-oss-120b     (flagship open-weight, tool-use capable)
    vision    : qwen/qwen3.6-27b        (multimodal; Groq serves it as preview)
    audio     : whisper-large-v3-turbo  (fast, cheap, generous free tier)

If Groq rotates these again, override without touching code:

    export GROQ_MODEL_ID="groq/openai/gpt-oss-120b"
    export GROQ_VISION_MODEL="groq/qwen/qwen3.6-27b"
    export GROQ_WHISPER_MODEL="whisper-large-v3-turbo"

All three read the same GROQ_API_KEY.
"""

import os
import re
import json
import time
import random

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    WikipediaSearchTool,
)
from smolagents.models import ChatMessage
from smolagents.monitoring import LogLevel

from tools import (
    read_file,
    analyze_image,
    transcribe_audio,
    get_youtube_transcript,
    visit_webpage,
)

# --------------------------------------------------------------------------- #
# CONFIG — everything provider-specific lives here                            #
# --------------------------------------------------------------------------- #
# LiteLLM routes any "groq/..." model id to Groq using GROQ_API_KEY.
# Qwen (qwen3.6-27b) is used for reasoning: it emits smolagents-style code blocks
# reliably, unlike Groq's gpt-oss models, which try to emit native tool calls and
# trip Groq's "Tool choice is none, but model called a tool" error under CodeAgent.
# It is also multimodal, so the same model powers analyze_image.
REASONING_MODEL_ID = os.getenv("GROQ_MODEL_ID", "groq/qwen/qwen3.6-27b")

# GAIA answer-format rules, appended to every task. The course grader does exact
# string matching, so format discipline is worth as much as being right.
GAIA_STRATEGY = """
Pick the approach that fits the question:
- Pure reasoning / math / logic / puzzles (e.g. a table to analyze, a sequence,
  unit conversions): DO NOT search the web. Write Python and compute the answer
  directly. You have pandas, numpy, math, statistics, itertools, collections.
- Facts about the world (people, dates, records, "how many X"): use web_search
  then visit_webpage to read the best source. For long pages (Wikipedia), if the
  section you need isn't in the first window, call visit_webpage(url,
  search="<keyword>") to jump to it.
- An attached file: use read_file for text/pdf/docx/pptx/csv; for spreadsheets
  you may also load it yourself, e.g. `import pandas as pd; df =
  pd.read_excel(path)`, then compute. Use analyze_image for images and
  transcribe_audio for audio.
- A YouTube link: use get_youtube_transcript.
Be efficient: most questions need 1–3 steps. Verify before answering.
""".strip()


GAIA_FORMATTING = """
When you have the answer, call final_answer with ONLY the answer itself — no
explanation, no "FINAL ANSWER" label, no surrounding sentence.

Format rules for the final answer:
- Number: no thousands separators, no units (no $, %, km) unless the question
  explicitly asks for the unit. Use plain Arabic digits.
- String: no leading articles (a/an/the), no abbreviations (spell out city and
  country names in full), match the exact spelling/casing of the source.
- List: comma-separated with a single space after each comma; apply the
  number/string rules to every element; do NOT wrap the list in brackets.
- If the question text itself is reversed/scrambled, answer in normal orientation.
""".strip()


# --------------------------------------------------------------------------- #
# Rate-limit resilient model                                                  #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# tool_use_failed recovery                                                    #
# --------------------------------------------------------------------------- #
# Some Groq models (notably the gpt-oss family) emit a NATIVE tool call even
# though CodeAgent runs with tool_choice=none, so Groq returns
#   400 tool_use_failed: "Tool choice is none, but model called a tool"
# The good news: the rejected payload is echoed back in a `failed_generation`
# field. We extract the code/tool-call the model wanted and re-present it as a
# normal CodeAgent code blob, so the step succeeds instead of dying.
def _extract_failed_generation(err_str: str) -> str | None:
    m = re.search(r'"failed_generation"\s*:\s*"(.*)"\s*\}\s*\}', err_str, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    try:
        raw = raw.encode("utf-8").decode("unicode_escape")
    except Exception:
        pass
    return raw


def _code_from_payload(payload: str) -> str:
    """payload ~ {"name": "<tool>", "arguments": <code-or-json>}."""
    payload = payload.strip()
    try:
        obj = json.loads(payload)
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if name in ("python", "code", "code_interpreter", "python_interpreter"):
            if isinstance(args, dict):
                return args.get("code") or args.get("arguments") or ""
            return str(args)
        # A real tool -> synthesize a python call to it (tools are py functions).
        if isinstance(args, dict):
            kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
            return f"result = {name}({kw})\nprint(result)"
        return f"result = {name}({args!r})\nprint(result)"
    except Exception:
        pass
    # Malformed JSON: take the raw code after "arguments":, drop the envelope brace.
    m = re.search(r'"arguments"\s*:\s*', payload)
    code = payload[m.end():] if m else payload
    return re.sub(r'\s*\}\s*$', '', code.strip()).strip()


def _recover_code_blob(err_str: str) -> str | None:
    payload = _extract_failed_generation(err_str)
    if not payload:
        return None
    code = _code_from_payload(payload)
    if not code.strip():
        return None
    # CodeAgent's parser looks for a ```py fenced block.
    return f"Thought: (recovered from a rejected tool call)\nCode:\n```py\n{code}\n```<end_code>"


class GroqLiteLLMModel(LiteLLMModel):
    """LiteLLMModel hardened for Groq under smolagents CodeAgent.

    Handles two Groq-specific failure modes:
    * Rate limits — the free/dev tiers cap tokens-per-minute (as low as ~6k),
      and a CodeAgent keeps the whole trajectory in context, so a multi-step
      question can trip the limit mid-run. We parse Groq's suggested wait
      ("try again in 4.2s") and sleep, with exponential backoff as a fallback.
    * tool_use_failed — a model tried to emit a native tool call; we recover the
      code from the error and return it as a valid code blob (see above).
    """

    def __init__(self, *args, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries

    @staticmethod
    def _parse_wait_seconds(err: Exception, attempt: int) -> float:
        msg = str(err)
        m = re.search(r"try again in ((\d+)m)?([\d.]+)s", msg, flags=re.IGNORECASE)
        if m:
            minutes = int(m.group(2)) if m.group(2) else 0
            return minutes * 60 + float(m.group(3))
        m = re.search(r'retry[-_ ]?(?:after|delay)"?[:=]\s*"?(\d+)', msg, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        return min(60.0, (2 ** attempt) + random.uniform(0, 1))

    def generate(self, messages, **kwargs):  # type: ignore[override]
        last_err = None
        for attempt in range(self._max_retries):
            try:
                return super().generate(messages, **kwargs)
            except Exception as e:
                s = str(e)
                # 1) Recover a rejected native tool call into a code blob.
                if "tool_use_failed" in s or "model called a tool" in s:
                    blob = _recover_code_blob(s)
                    if blob:
                        print("[groq] recovered code from a rejected tool call")
                        return ChatMessage(role="assistant", content=blob)
                    raise
                # 2) Back off on rate limits.
                if "rate" in s.lower() or "429" in s:
                    last_err = e
                    wait = self._parse_wait_seconds(e, attempt)
                    print(f"[groq] rate limited (attempt {attempt + 1}/"
                          f"{self._max_retries}); sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise  # anything else: surface immediately
        raise RuntimeError(f"Groq rate limit not cleared after "
                           f"{self._max_retries} retries: {last_err}")


def build_model() -> GroqLiteLLMModel:
    """Reasoning model. temperature=0 for deterministic, graded answers."""
    if not os.getenv("GROQ_API_KEY"):
        print("[warn] GROQ_API_KEY is not set — Groq calls will fail.")
    return GroqLiteLLMModel(model_id=REASONING_MODEL_ID, temperature=0.0)


# --------------------------------------------------------------------------- #
# Agent construction                                                          #
# --------------------------------------------------------------------------- #
def build_agent(verbose: bool = False) -> CodeAgent:
    model = build_model()
    tools = [
        DuckDuckGoSearchTool(),
        visit_webpage,
        # summary mode keeps each Wikipedia hit small enough for Groq's tight
        # per-minute token cap (full-text mode was ~7k tokens per call).
        WikipediaSearchTool(content_type="summary"),
        read_file,
        analyze_image,      # -> Groq vision
        transcribe_audio,   # -> Groq Whisper
        get_youtube_transcript,
    ]
    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=6,
        verbosity_level=LogLevel.DEBUG if verbose else LogLevel.INFO,
        additional_authorized_imports=[
            "pandas", "numpy", "math", "statistics", "datetime",
            "json", "re", "itertools", "collections",
            "openpyxl", "csv", "docx", "pptx", "bs4",
        ],
    )


# --------------------------------------------------------------------------- #
# Trace + answer normalization                                                #
# --------------------------------------------------------------------------- #
def format_trace(agent: CodeAgent) -> str:
    """Compact post-run summary of the LAST question, for --trace / logs."""
    lines = []
    for step in getattr(agent.memory, "steps", []):
        n = getattr(step, "step_number", None)
        if n is None:
            continue
        lines.append(f"--- step {n} ---")
        for tc in (getattr(step, "tool_calls", None) or []):
            lines.append(f"  call: {tc.name}({tc.arguments})")
        code = getattr(step, "code_action", None)
        if code:
            lines.append("  code: " + code.strip().replace("\n", "\n        "))
        obs = getattr(step, "observations", None)
        if obs:
            obs = obs.strip()
            lines.append("  obs : " + (obs[:600] + ("  ...[truncated]" if len(obs) > 600 else "")))
        err = getattr(step, "error", None)
        if err:
            lines.append(f"  ERR : {err}")
        out = getattr(step, "action_output", None)
        if out is not None:
            lines.append(f"  out : {out!r}")
    return "\n".join(lines) if lines else "(no steps recorded)"


_REVERSED_HINT = re.compile(r"\.(rewsna|drow|ecnetnes)", re.IGNORECASE)


def looks_reversed(text: str) -> bool:
    """Heuristic for GAIA's reversed-text question (starts with '.' and reads
    backwards). Cheap enough to check every question."""
    if not text:
        return False
    t = text.strip()
    return t.startswith(".") or bool(_REVERSED_HINT.search(t))


def normalize_answer(ans: str) -> str:
    """Post-process the model's final answer toward GAIA exact-match form."""
    if ans is None:
        return ""
    s = str(ans).strip()

    # Strip a stray "FINAL ANSWER:" / "Answer:" prefix if the model added one.
    s = re.sub(r"^\s*(final answer|answer)\s*:?\s*", "", s, flags=re.IGNORECASE)

    # Strip surrounding quotes and a trailing period (but not a decimal point).
    s = s.strip().strip('"').strip("'").strip()
    if s.endswith(".") and not re.search(r"\d\.\d", s):
        s = s[:-1].strip()

    # If the whole thing is a single number, drop separators / currency / percent.
    bare = s.replace(",", "").replace("$", "").replace("%", "").strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", bare):
        return bare

    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------- #
# Public wrapper (interface preserved for app.py / run_local.py)              #
# --------------------------------------------------------------------------- #
class GaiaAgent:
    """Callable used by app.py. One instance answers all questions."""

    def __init__(self, verbose: bool = False):
        self.agent = build_agent(verbose=verbose)
        self.last_trace = ""
        print(f"GaiaAgent ready on Groq ({REASONING_MODEL_ID}).")

    def __call__(self, question: str, file_path: str | None = None) -> str:
        task = question
        if looks_reversed(question):
            task += (
                "\n\nNOTE: the question text above appears to be reversed. "
                f"Reversed reading:\n{question[::-1]}"
            )
        if file_path:
            task += (
                f"\n\nA file is attached at the local path:\n{file_path}\n"
                "Use read_file / analyze_image / transcribe_audio on that path "
                "as appropriate."
            )
        task += "\n\n" + GAIA_STRATEGY + "\n\n" + GAIA_FORMATTING

        try:
            raw = self.agent.run(task)
        except Exception as e:
            print(f"[agent error] {type(e).__name__}: {e}")
            self.last_trace = format_trace(self.agent)
            return "ERROR"
        self.last_trace = format_trace(self.agent)
        return normalize_answer(raw)
