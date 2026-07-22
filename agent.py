"""
GaiaAgent — smolagents CodeAgent for the HF Agents Course GAIA subset.

Provider-flexible (Gemini or Groq), tuned for a final scoring run:
  * never wastes steps on questions whose attachment the scoring API does not
    serve (the API returns "No file path associated with task_id" for all five
    file questions — they are unanswerable for everyone),
  * recovers from empty / malformed generations instead of losing the question,
  * salvages a printed answer if the agent computed it but died before calling
    final_answer.

Provider selection (first match wins):
    AGENT_MODEL_ID   explicit LiteLLM model id, e.g. "gemini/gemini-2.5-flash"
    GEMINI_API_KEY   -> gemini/gemini-2.5-flash   (recommended: high free RPD,
                        native vision, far more headroom than Groq's free TPM)
    GROQ_API_KEY     -> groq/qwen/qwen3.6-27b

Other env knobs:
    AGENT_VISION_MODEL   image model (defaults to the reasoning model)
    GROQ_WHISPER_MODEL   audio model (Groq Whisper; needs GROQ_API_KEY)
    GAIA_MAX_STEPS               default 6
    GAIA_MAX_QUESTION_SLEEP      default 180 (seconds of rate-limit sleep per Q)
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
# Provider / model selection                                                  #
# --------------------------------------------------------------------------- #
# Google retires Gemini model ids fast, and free-tier quotas differ enormously
# BY MODEL. The binding limit on the free tier is requests-per-day, not tokens:
#   gemini-3.1-flash-lite : 15 RPM, 250K TPM, 500 RPD  <- only one that can
#                                                          finish a full run
#   gemini-3.6-flash      :  5 RPM, 250K TPM,  20 RPD  <- runs out after ~5 Qs
#   gemini-3-flash        :  5 RPM, 250K TPM,  20 RPD
# So flash-lite is the default; set AGENT_MODEL_ID to override (e.g. if you
# have billing enabled, gemini/gemini-3.6-flash is the stronger model).
# Check https://ai.google.dev/gemini-api/docs/models if these 404.
GEMINI_FALLBACKS = [
    "gemini/gemini-3.1-flash-lite",  # best free-tier quota (500 RPD)
    "gemini/gemini-3.6-flash",       # stronger, but 20 RPD free
    "gemini/gemini-3-flash",
    "gemini/gemini-flash-latest",
]


def _resolve_model_id() -> str:
    explicit = os.getenv("AGENT_MODEL_ID") or os.getenv("GROQ_MODEL_ID")
    if explicit:
        return explicit
    if os.getenv("GEMINI_API_KEY"):
        return GEMINI_FALLBACKS[0]
    return "groq/qwen/qwen3.6-27b"


MODEL_ID = _resolve_model_id()
VISION_MODEL_ID = os.getenv("AGENT_VISION_MODEL", MODEL_ID)

MAX_QUESTION_SLEEP = float(os.getenv("GAIA_MAX_QUESTION_SLEEP", "180"))
MAX_STEPS = int(os.getenv("GAIA_MAX_STEPS", "6"))

GAIA_STRATEGY = (
    "Answer in as few steps as possible.\n"
    "- Math/logic/table/puzzle: write Python and compute it. Do NOT search.\n"
    "- Facts about the world: web_search, then visit_webpage on the best "
    "result. On a long page use visit_webpage(url, search='<keyword>') to jump "
    "to the right section instead of re-fetching.\n"
    "- Multi-step fact questions: solve ONE hop per step and print what you "
    "found, then use it in the next step.\n"
    "- YouTube: get_youtube_transcript.\n"
    "- Encoded or reversed text: decode it in Python.\n"
    "If a tool fails twice, stop using it and answer with your best estimate. "
    "As soon as you know the answer, call final_answer immediately."
)

GAIA_FORMATTING = (
    "Every step MUST contain a code block. Keep reasoning to one or two short "
    "sentences before the code — never reason without emitting code.\n"
    "Call final_answer with ONLY the answer: no label, no sentence. "
    "Number: plain digits, no separators or units unless asked. "
    "String: no articles, no abbreviations, match the source spelling. "
    "List: comma+space separated, no brackets."
)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _strip_think(text):
    """Remove <think>...</think> reasoning blocks — but NEVER return an empty
    string when the input was non-empty. A model that emits only a reasoning
    block would otherwise produce an empty message, which smolagents reports as
    'Error in generating model output:' and the question is lost."""
    if not isinstance(text, str) or not text.strip():
        return text
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in out and "</think>" not in out:
        head = out.split("<think>", 1)[0]
        if head.strip():
            out = head
    if not out.strip():
        return text          # stripping would empty it -> keep the original
    return out.strip()


def _is_empty_message(msg) -> bool:
    if getattr(msg, "tool_calls", None):
        return False
    content = getattr(msg, "content", None)
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    return not content


def _extract_failed_generation(err_str: str):
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
    return re.sub(r"\s*\}\s*$", "", code.strip()).strip()


def _recover_code_blob(err_str: str):
    payload = _extract_failed_generation(err_str)
    if not payload:
        return None
    code = _code_from_payload(payload)
    if not code.strip():
        return None
    # <code> tags are what the current smolagents parser looks for first.
    return ("Thought: (recovered from a rejected tool call)\n"
            f"<code>\n{code}\n</code>")


# --------------------------------------------------------------------------- #
# Model wrapper                                                               #
# --------------------------------------------------------------------------- #
class RobustLiteLLMModel(LiteLLMModel):
    """Hardened model wrapper.

    Handles the three failure modes seen in real runs:
      1. rate limits          -> parse the provider's suggested wait and sleep,
                                 bounded by a per-question budget
      2. tool_use_failed      -> recover the rejected code and run it
      3. empty generation     -> retry the call instead of losing the question
    """

    def __init__(self, *args, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries
        self._question_sleep = 0.0

    def new_question(self):
        self._question_sleep = 0.0

    @staticmethod
    def _parse_wait_seconds(err: Exception, attempt: int) -> float:
        msg = str(err)
        m = re.search(r"try again in ((\d+)m)?([\d.]+)s", msg, flags=re.IGNORECASE)
        if m:
            minutes = int(m.group(2)) if m.group(2) else 0
            return minutes * 60 + float(m.group(3))
        m = re.search(r'retry[-_ ]?(?:after|delay)"?[:=]\s*"?(\d+)',
                      msg, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        return min(45.0, (2 ** attempt) + random.uniform(0, 1))

    def _try_next_model(self) -> bool:
        """Swap to the next known-good model id after a deprecation 404."""
        if not self.model_id.startswith("gemini/"):
            return False
        try:
            idx = GEMINI_FALLBACKS.index(self.model_id)
        except ValueError:
            idx = -1
        if idx + 1 >= len(GEMINI_FALLBACKS):
            return False
        new_id = GEMINI_FALLBACKS[idx + 1]
        print(f"[model] {self.model_id} unavailable -> switching to {new_id}")
        self.model_id = new_id
        return True

    def generate(self, messages, **kwargs):  # type: ignore[override]
        last_err = None
        for attempt in range(self._max_retries):
            try:
                msg = super().generate(messages, **kwargs)
                if getattr(msg, "content", None):
                    msg.content = _strip_think(msg.content)
                if _is_empty_message(msg):
                    # Model returned nothing usable. Retry rather than lose the
                    # whole question to "Error in generating model output:".
                    print(f"[model] empty generation "
                          f"(attempt {attempt + 1}/{self._max_retries}); retrying")
                    time.sleep(1.0 + attempt)
                    continue
                return msg
            except Exception as e:
                s = str(e)
                if "tool_use_failed" in s or "model called a tool" in s:
                    blob = _recover_code_blob(s)
                    if blob:
                        print("[model] recovered code from a rejected tool call")
                        return ChatMessage(role="assistant", content=blob)
                    raise
                if ("404" in s or "NOT_FOUND" in s or "no longer available" in s
                        or "NotFoundError" in type(e).__name__):
                    last_err = e
                    if self._try_next_model():
                        continue
                    raise RuntimeError(
                        "no usable Gemini model id; check "
                        "https://ai.google.dev/gemini-api/docs/models and set "
                        f"AGENT_MODEL_ID. Last error: {e}")
                if "rate" in s.lower() or "429" in s:
                    last_err = e
                    wait = self._parse_wait_seconds(e, attempt)
                    if self._question_sleep + wait > MAX_QUESTION_SLEEP:
                        raise RuntimeError(
                            f"per-question sleep budget "
                            f"({MAX_QUESTION_SLEEP:.0f}s) exhausted; skipping")
                    self._question_sleep += wait
                    print(f"[model] rate limited "
                          f"(attempt {attempt + 1}/{self._max_retries}); "
                          f"sleeping {wait:.1f}s "
                          f"(budget {self._question_sleep:.0f}/"
                          f"{MAX_QUESTION_SLEEP:.0f}s)")
                    time.sleep(wait)
                    continue
                last_err = e
                raise
        # Retries exhausted.
        if last_err:
            raise RuntimeError(f"generation failed after {self._max_retries} "
                               f"attempts: {last_err}")
        # All attempts returned empty -> emit a nudge so the step still runs.
        print("[model] all attempts empty; nudging agent to emit code")
        return ChatMessage(
            role="assistant",
            content=("Thought: I must emit code.\n<code>\n"
                     "print('retry: emit an answer now')\n</code>"),
        )


def build_model() -> RobustLiteLLMModel:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")):
        print("[warn] no GEMINI_API_KEY or GROQ_API_KEY set — calls will fail.")
    if "gpt-oss" in MODEL_ID:
        print("[warn] gpt-oss breaks CodeAgent (tool_use_failed / bad code "
              "blobs). Prefer gemini/gemini-2.5-flash or groq/qwen/qwen3.6-27b.")
    return RobustLiteLLMModel(model_id=MODEL_ID, temperature=0.0)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
def build_agent(verbose: bool = False) -> CodeAgent:
    tools = [
        DuckDuckGoSearchTool(),
        visit_webpage,
        WikipediaSearchTool(content_type="summary"),
        get_youtube_transcript,
        read_file,
        analyze_image,
        transcribe_audio,
    ]
    return CodeAgent(
        tools=tools,
        model=build_model(),
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
# Trace, salvage, normalization                                               #
# --------------------------------------------------------------------------- #
def format_trace(agent: CodeAgent) -> str:
    lines = []
    for step in getattr(agent.memory, "steps", []):
        n = getattr(step, "step_number", None)
        if n is None:
            continue
        lines.append(f"--- step {n} ---")
        code = getattr(step, "code_action", None)
        if code:
            lines.append("  code: " + code.strip().replace("\n", "\n        "))
        obs = getattr(step, "observations", None)
        if obs:
            obs = obs.strip()
            lines.append("  obs : " + (obs[:600] +
                         ("  ...[truncated]" if len(obs) > 600 else "")))
        err = getattr(step, "error", None)
        if err:
            lines.append(f"  ERR : {err}")
        out = getattr(step, "action_output", None)
        if out is not None:
            lines.append(f"  out : {out!r}")
    return "\n".join(lines) if lines else "(no steps recorded)"


_BAD_OBS = re.compile(r"error|traceback|not found|failed|no wikipedia|"
                      r"search results|\[window around|\[first \d+ of",
                      re.IGNORECASE)


def salvage_answer(agent: CodeAgent):
    """If the agent computed the answer and printed it but died before calling
    final_answer, recover that printed value.

    Deliberately conservative: only short, clean, single-line observations from
    the most recent steps qualify — never error text, search dumps, or page
    windows.
    """
    steps = list(getattr(agent.memory, "steps", []))
    for step in reversed(steps):
        obs = getattr(step, "observations", None)
        if not obs:
            continue
        text = obs.strip()
        # smolagents prefixes execution output; keep only the printed part.
        text = re.sub(r"^Execution logs:\s*", "", text).strip()
        text = re.sub(r"\bOut:\s*None\s*$", "", text).strip()
        if not text or len(text) > 160 or "\n" in text:
            continue
        if _BAD_OBS.search(text):
            continue
        return text
    return None


_REVERSED_HINT = re.compile(r"\.(rewsna|drow|ecnetnes)", re.IGNORECASE)


def looks_reversed(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return t.startswith(".") or bool(_REVERSED_HINT.search(t))


def normalize_answer(ans) -> str:
    if ans is None:
        return ""
    s = _strip_think(str(ans)) or ""
    s = s.strip()
    s = re.sub(r"^\s*(final answer|answer)\s*:?\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("```py", "").replace("```python", "").replace("```", "")
    s = s.replace("<code>", "").replace("</code>", "").replace("<end_code>", "")
    s = s.replace("<think>", "").replace("</think>", "")
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
        print(f"GaiaAgent ready ({MODEL_ID}).")

    def __call__(self, question: str, file_path: str | None = None) -> str:
        try:
            self.agent.model.new_question()
        except Exception:
            pass

        task = question
        if looks_reversed(question):
            task += ("\n\nNOTE: the question text may be reversed. Reversed "
                     f"reading:\n{question[::-1]}")
        if file_path:
            task += (f"\n\nA file is attached at:\n{file_path}\n"
                     "Use read_file / analyze_image / transcribe_audio on it.")
        task += "\n\n" + GAIA_STRATEGY + "\n\n" + GAIA_FORMATTING

        raw = None
        try:
            raw = self.agent.run(task)
        except Exception as e:
            print(f"[agent error] {type(e).__name__}: {e}")

        self.last_trace = format_trace(self.agent)
        answer = normalize_answer(raw) if raw is not None else ""

        # If the run produced nothing usable, try to recover a printed answer.
        if not answer or answer.upper() == "ERROR":
            rescued = salvage_answer(self.agent)
            if rescued:
                print(f"[salvage] recovered printed answer: {rescued!r}")
                return normalize_answer(rescued)
            return "ERROR"
        return answer
