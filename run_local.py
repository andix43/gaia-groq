"""
Local / Colab runner for the Groq GAIA agent. Designed to complete all 20
questions smoothly: each question has a HARD wall-clock timeout, and any
failure is recorded as "ERROR" so the run never aborts partway.

Develop (no submission):
    python run_local.py --refresh              # cache 20 questions, run all
    python run_local.py --id 0 --trace         # one question, full trace
    python run_local.py --timeout 240          # cap each question at 240s

Submit (scores you; no running Space required):
    python run_local.py --submit \
        --username <you> \
        --agent-code https://huggingface.co/spaces/<you>/<space>/tree/main
Requires GROQ_API_KEY in the environment.
"""

import argparse
import json
import os
import signal
import tempfile
import time

import requests

from agent import GaiaAgent

API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE = "questions_cache.json"


# --------------------------------------------------------------------------- #
# hard per-question timeout (Unix / Colab)                                    #
# --------------------------------------------------------------------------- #
class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


_HAS_ALARM = hasattr(signal, "SIGALRM")
if _HAS_ALARM:
    signal.signal(signal.SIGALRM, _alarm_handler)


# --------------------------------------------------------------------------- #
def load_questions(refresh: bool):
    if refresh or not os.path.exists(CACHE):
        r = requests.get(f"{API_URL}/questions", timeout=30)
        r.raise_for_status()
        with open(CACHE, "w") as f:
            json.dump(r.json(), f, indent=2)
    with open(CACHE) as f:
        return json.load(f)


def download(task_id, file_name):
    try:
        r = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"[file] no file for {task_id} ({e})")
        return None
    suffix = os.path.splitext(file_name)[1] if file_name else ""
    path = os.path.join(tempfile.gettempdir(), f"{task_id}{suffix}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def submit(username, agent_code, answers_payload):
    payload = {"username": username.strip(), "agent_code": agent_code,
               "answers": answers_payload}
    r = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def answer_one(agent, question, file_path, timeout):
    """Run the agent on one question with a hard timeout. Returns a string."""
    if _HAS_ALARM and timeout > 0:
        signal.alarm(int(timeout))
    try:
        return agent(question, file_path)
    except _Timeout:
        print(f"[timeout] question exceeded {timeout}s; recording ERROR")
        return "ERROR"
    finally:
        if _HAS_ALARM:
            signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--id", type=int, default=None, help="run only this index")
    ap.add_argument("--trace", action="store_true", help="print + save full traces")
    ap.add_argument("--timeout", type=int, default=300,
                    help="hard cap (seconds) per question; 0 disables")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds to wait between questions (eases rate limits)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--username", default=os.getenv("HF_USERNAME"))
    ap.add_argument("--agent-code", default=os.getenv("AGENT_CODE"))
    ap.add_argument("--space", default=os.getenv("HF_SPACE_ID"),
                    help="HF space id 'user/space'; becomes the agent_code URL")
    args = ap.parse_args()

    agent_code = args.agent_code
    if not agent_code and args.space:
        agent_code = f"https://huggingface.co/spaces/{args.space}/tree/main"
    if args.submit and (not args.username or not agent_code):
        raise SystemExit("--submit needs --username and (--agent-code URL or --space id).")

    questions = load_questions(args.refresh)
    agent = GaiaAgent(verbose=args.trace)
    if args.trace:
        os.makedirs("traces", exist_ok=True)

    answers_payload = []
    t0 = time.time()
    for i, item in enumerate(questions):
        if args.id is not None and i != args.id:
            continue
        task_id = item["task_id"]
        file_name = item.get("file_name") or ""

        print("\n" + "=" * 80)
        print(f"[{i}] {task_id}  file={file_name or '-'}")
        print("Q:", item["question"])

        try:
            file_path = download(task_id, file_name) if file_name else None
            ans = answer_one(agent, item["question"], file_path, args.timeout)
        except KeyboardInterrupt:
            print("\n[interrupted] keeping answers so far.")
            break
        except Exception as e:
            print(f"[question {i} failed] {type(e).__name__}: {e}")
            ans = "ERROR"

        print("A:", repr(ans))
        answers_payload.append({"task_id": task_id, "submitted_answer": ans})

        if args.trace:
            trace_path = os.path.join("traces", f"{i:02d}_{task_id}.txt")
            with open(trace_path, "w") as f:
                f.write(f"Q: {item['question']}\nA: {ans!r}\n\n{agent.last_trace}\n")
            print("TRACE:\n" + agent.last_trace + f"\n(saved -> {trace_path})")

        if args.pace and (args.id is None):
            time.sleep(args.pace)

    print(f"\nFinished {len(answers_payload)} question(s) in "
          f"{time.time() - t0:.0f}s.")

    if args.submit:
        print("Submitting", len(answers_payload), "answers...")
        result = submit(args.username, agent_code, answers_payload)
        print(json.dumps(result, indent=2))
        print(f"\nScore: {result.get('score')}%  "
              f"({result.get('correct_count')}/{result.get('total_attempted')})")


if __name__ == "__main__":
    main()
