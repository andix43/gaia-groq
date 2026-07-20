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
import time
import random

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    WikipediaSearchTool,
)
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
REASONING_MODEL_ID = os.getenv("GROQ_MODEL_ID", "groq/openai/gpt-oss-120b")

# GAIA answer-format rules, appended to every task. The course grader does exact
# string matching, so format discipline is worth as much as being right.
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
class GroqLiteLLMModel(LiteLLMModel):
    """LiteLLMModel that retries on Groq's per-minute rate limits.

    Groq's free/dev tiers cap tokens-per-minute (as low as ~6k TPM), and a
    CodeAgent keeps the whole trajectory in context, so a multi-step question
    can trip the limit mid-run. On a RateLimitError we parse Groq's suggested
    wait ("Please try again in 4.2s" / a ``retry-after`` header) and sleep that
    long, with exponential backoff + jitter as a fallback.
    """

    def __init__(self, *args, max_retries: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries

    @staticmethod
    def _parse_wait_seconds(err: Exception, attempt: int) -> float:
        msg = str(err)
        # Groq phrasings: "try again in 4.2s" / "in 1m2.3s"
        m = re.search(r"try again in ((\d+)m)?([\d.]+)s", msg, flags=re.IGNORECASE)
        if m:
            minutes = int(m.group(2)) if m.group(2) else 0
            return minutes * 60 + float(m.group(3))
        # retryDelay JSON (some gateways) or Retry-After header echoed in text
        m = re.search(r'retry[-_ ]?(?:after|delay)"?[:=]\s*"?(\d+)', msg, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        # Fallback: exponential backoff with jitter, capped at 60s.
        return min(60.0, (2 ** attempt) + random.uniform(0, 1))

    def generate(self, messages, **kwargs):  # type: ignore[override]
        last_err = None
        for attempt in range(self._max_retries):
            try:
                return super().generate(messages, **kwargs)
            except Exception as e:  # litellm.RateLimitError and friends
                if "rate" not in str(e).lower() and "429" not in str(e):
                    raise  # not a rate-limit problem — surface immediately
                last_err = e
                wait = self._parse_wait_seconds(e, attempt)
                print(f"[groq] rate limited (attempt {attempt + 1}/"
                      f"{self._max_retries}); sleeping {wait:.1f}s")
                time.sleep(wait)
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
        task += "\n\n" + GAIA_FORMATTING

        try:
            raw = self.agent.run(task)
        except Exception as e:
            print(f"[agent error] {type(e).__name__}: {e}")
            self.last_trace = format_trace(self.agent)
            return "ERROR"
        self.last_trace = format_trace(self.agent)
        return normalize_answer(raw)
