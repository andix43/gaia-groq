"""
GaiaAgent — a smolagents CodeAgent for the HF Agents Course GAIA subset, running
entirely on Groq. Final build: tuned to run through all 20 questions smoothly
(bounded time + steps per question) rather than to maximize accuracy.

Models (all via GROQ_API_KEY, overridable by env):
    GROQ_MODEL_ID      reasoning + vision   default groq/qwen/qwen3.6-27b
    GROQ_VISION_MODEL  images               default = GROQ_MODEL_ID
    GROQ_WHISPER_MODEL audio                default whisper-large-v3-turbo

Robustness features:
  * Rate-limit backoff with a PER-QUESTION sleep budget (bails a question
    gracefully instead of blocking the whole run).
  * tool_use_failed recovery (recovers code from a rejected native tool call).
  * <think>-block stripping (keeps context small; stops reasoning leaking into
    answers).
  * Answer normalization toward GAIA exact-match format.
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
    analyze_youtube_video,
    visit_webpage,
)

# --------------------------------------------------------------------------- #
# CONFIG                                                                      #
# --------------------------------------------------------------------------- #
REASONING_MODEL_ID = os.getenv("GROQ_MODEL_ID", "groq/qwen/qwen3.6-27b")

# Max seconds a single question may spend sleeping on rate limits before we give
# up on it and move on. Keeps the whole 20-question run bounded.
MAX_QUESTION_SLEEP = float(os.getenv("GAIA_MAX_QUESTION_SLEEP", "150"))
MAX_STEPS = int(os.getenv("GAIA_MAX_STEPS", "4"))

GAIA_STRATEGY = (
    "Choose the fewest steps. Pure math/logic/table/puzzle: write Python and "
    "compute (pandas, numpy, math, itertools) — do NOT search. World facts: "
    "web_search then visit_webpage; for a long page, visit_webpage(url, "
    "search='<keyword>') jumps to a section. Attached file: read_file "
    "(text/pdf/docx/pptx/csv/xlsx) or load with pandas; analyze_image for "
    "images; transcribe_audio for audio. YouTube: get_youtube_transcript "
    "(spoken) or analyze_youtube_video (visual). Encoded/reversed text: decode "
    "in Python. If a tool fails twice, give your best answer instead of "
    "retrying. Verify, then answer."
)

GAIA_FORMATTING = (
    "Call final_answer with ONLY the answer — no label, no sentence. Number: "
    "plain digits, no separators/units unless asked. String: no articles, no "
    "abbreviations, match the source's spelling. List: comma+space separated, "
    "no brackets."
)


# --------------------------------------------------------------------------- #
# helpers: think-stripping + tool_use_failed recovery                         #
# --------------------------------------------------------------------------- #
def _strip_think(text):
    if not isinstance(text, str):
        return text
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in out and "</think>" not in out:
        head = out.split("<think>", 1)[0]
        out = head if head.strip() else out.replace("<think>", "")
    return out.strip()


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
    payload = payload.strip()
    try:
        obj = json.loads(payload)
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if name in ("python", "code", "code_interpreter", "python_interpreter"):
            if isinstance(args, dict):
                return args.get("code") or args.get("arguments") or ""
            return str(args)
        if isinstance(args, dict):
            kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
            return f"result = {name}({kw})\nprint(result)"
        return f"result = {name}({args!r})\nprint(result)"
    except Exception:
        pass
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
    return f"Thought: (recovered from a rejected tool call)\nCode:\n```py\n{code}\n```<end_code>"


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class GroqLiteLLMModel(LiteLLMModel):
    """Groq-hardened model: rate-limit backoff (with per-question sleep budget),
    tool_use_failed recovery, and <think> stripping."""

    def __init__(self, *args, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries
        self._question_sleep = 0.0

    def new_question(self):
        """Reset the per-question sleep budget. Call at the start of each Q."""
        self._question_sleep = 0.0

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
                msg = super().generate(messages, **kwargs)
                if getattr(msg, "content", None):
                    msg.content = _strip_think(msg.content)
                return msg
            except Exception as e:
                s = str(e)
                if "tool_use_failed" in s or "model called a tool" in s:
                    blob = _recover_code_blob(s)
                    if blob:
                        print("[groq] recovered code from a rejected tool call")
                        return ChatMessage(role="assistant", content=blob)
                    raise
                if "rate" in s.lower() or "429" in s:
                    last_err = e
                    wait = self._parse_wait_seconds(e, attempt)
                    if self._question_sleep + wait > MAX_QUESTION_SLEEP:
                        raise RuntimeError(
                            f"per-question sleep budget "
                            f"({MAX_QUESTION_SLEEP:.0f}s) exhausted; skipping"
                        )
                    self._question_sleep += wait
                    print(f"[groq] rate limited (attempt {attempt + 1}/"
                          f"{self._max_retries}); sleeping {wait:.1f}s "
                          f"(budget {self._question_sleep:.0f}/"
                          f"{MAX_QUESTION_SLEEP:.0f}s)")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"rate limit not cleared after {self._max_retries} "
                           f"retries: {last_err}")


def build_model() -> GroqLiteLLMModel:
    if not os.getenv("GROQ_API_KEY"):
        print("[warn] GROQ_API_KEY is not set — Groq calls will fail.")
    if "gpt-oss" in REASONING_MODEL_ID:
        print("[warn] gpt-oss breaks CodeAgent (tool_use_failed / bad code "
              "blobs). Prefer groq/qwen/qwen3.6-27b; unset GROQ_MODEL_ID.")
    return GroqLiteLLMModel(model_id=REASONING_MODEL_ID, temperature=0.0)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
def build_agent(verbose: bool = False) -> CodeAgent:
    model = build_model()
    tools = [
        DuckDuckGoSearchTool(),
        visit_webpage,
        WikipediaSearchTool(content_type="summary"),
        read_file,
        analyze_image,
        transcribe_audio,
        get_youtube_transcript,
        analyze_youtube_video,
    ]
    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=MAX_STEPS,
        verbosity_level=LogLevel.DEBUG if verbose else LogLevel.INFO,
        additional_authorized_imports=[
            "pandas", "numpy", "math", "statistics", "datetime",
            "json", "re", "itertools", "collections",
            "openpyxl", "csv", "docx", "pptx", "bs4",
            "base64", "codecs", "urllib", "fractions", "decimal",
        ],
    )


# --------------------------------------------------------------------------- #
# Trace + normalization                                                       #
# --------------------------------------------------------------------------- #
def format_trace(agent: CodeAgent) -> str:
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
    if not text:
        return False
    t = text.strip()
    return t.startswith(".") or bool(_REVERSED_HINT.search(t))


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    s = _strip_think(str(ans)).strip()
    s = re.sub(r"^\s*(final answer|answer)\s*:?\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("```py", "").replace("```python", "").replace("```", "")
    s = s.replace("<code>", "").replace("</code>", "").replace("<end_code>", "")
    s = s.strip().strip('"').strip("'").strip()
    if s.endswith(".") and not re.search(r"\d\.\d", s):
        s = s[:-1].strip()
    bare = s.replace(",", "").replace("$", "").replace("%", "").strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", bare):
        return bare
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------- #
# Public wrapper                                                              #
# --------------------------------------------------------------------------- #
class GaiaAgent:
    def __init__(self, verbose: bool = False):
        self.agent = build_agent(verbose=verbose)
        self.last_trace = ""
        print(f"GaiaAgent ready on Groq ({REASONING_MODEL_ID}).")

    def __call__(self, question: str, file_path: str | None = None) -> str:
        # Reset the per-question rate-limit sleep budget.
        try:
            self.agent.model.new_question()
        except Exception:
            pass

        task = question
        if looks_reversed(question):
            task += ("\n\nNOTE: the question text may be reversed. Reversed "
                     f"reading:\n{question[::-1]}")
        if file_path:
            task += (f"\n\nA file is attached at:\n{file_path}\nUse read_file / "
                     "analyze_image / transcribe_audio on it as appropriate.")
        task += "\n\n" + GAIA_STRATEGY + "\n\n" + GAIA_FORMATTING + "\n/no_think"

        try:
            raw = self.agent.run(task)
        except Exception as e:
            print(f"[agent error] {type(e).__name__}: {e}")
            self.last_trace = format_trace(self.agent)
            return "ERROR"
        self.last_trace = format_trace(self.agent)
        return normalize_answer(raw)
