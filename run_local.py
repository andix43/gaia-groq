"""
Runner for the GAIA agent. Final-run behaviour:

  * Questions whose attachment the scoring API refuses to serve are SKIPPED
    instantly (no model calls, no wasted rate-limit budget). The API returns
    "No file path associated with task_id" for all five file questions, so they
    are unanswerable for everyone; spending steps on them only starves the
    questions you can win.
  * Every answerable question gets a hard wall-clock timeout.
  * Any failure is recorded as ERROR; the run never aborts partway.

Usage:
    python run_local.py --refresh                 # run all answerable questions
    python run_local.py --id 0 --trace            # one question, full trace
    python run_local.py --refresh --timeout 240
    python run_local.py --submit --username <you> \
        --agent-code https://huggingface.co/spaces/<you>/<space>/tree/main
"""

import argparse
import json
import os
import signal
import tempfile
import time

import requests

from agent import GaiaAgent, RequestBudgetExceeded

API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE = "questions_cache.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GaiaAgent/1.0)"}


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


_HAS_ALARM = hasattr(signal, "SIGALRM")
if _HAS_ALARM:
    signal.signal(signal.SIGALRM, _alarm_handler)


def load_questions(refresh: bool):
    if refresh or not os.path.exists(CACHE):
        r = requests.get(f"{API_URL}/questions", timeout=30)
        r.raise_for_status()
        with open(CACHE, "w") as f:
            json.dump(r.json(), f, indent=2)
    with open(CACHE) as f:
        return json.load(f)


def download(task_id, file_name):
    """Return a local path, or None if the API will not serve the file."""
    try:
        r = requests.get(f"{API_URL}/files/{task_id}", headers=HEADERS, timeout=60)
        if r.status_code != 200 or not r.content:
            print(f"[file] not served ({r.status_code}) for {task_id}")
            return None
    except Exception as e:
        print(f"[file] download failed for {task_id}: {e}")
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
    if _HAS_ALARM and timeout > 0:
        signal.alarm(int(timeout))
    try:
        return agent(question, file_path)
    except _Timeout:
        print(f"[timeout] exceeded {timeout}s; recording ERROR")
        return "ERROR"
    finally:
        if _HAS_ALARM:
            signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--id", type=int, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--timeout", type=int, default=300,
                    help="hard cap (seconds) per question; 0 disables")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds between questions (eases rate limits)")
    ap.add_argument("--try-files", action="store_true",
                    help="attempt file questions even if the API serves no file")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--username", default=os.getenv("HF_USERNAME"))
    ap.add_argument("--agent-code", default=os.getenv("AGENT_CODE"))
    ap.add_argument("--space", default=os.getenv("HF_SPACE_ID"))
    args = ap.parse_args()

    agent_code = args.agent_code
    if not agent_code and args.space:
        agent_code = f"https://huggingface.co/spaces/{args.space}/tree/main"
    if args.submit and (not args.username or not agent_code):
        raise SystemExit("--submit needs --username and (--agent-code or --space).")

    questions = load_questions(args.refresh)
    agent = GaiaAgent(verbose=args.trace)
    if args.trace:
        os.makedirs("traces", exist_ok=True)

    answers_payload, results = [], []
    skipped = 0
    t0 = time.time()

    for i, item in enumerate(questions):
        if args.id is not None and i != args.id:
            continue
        task_id = item["task_id"]
        file_name = item.get("file_name") or ""

        print("\n" + "=" * 80)
        print(f"[{i}] {task_id}  file={file_name or '-'}")
        print("Q:", item["question"][:200])

        try:
            file_path = download(task_id, file_name) if file_name else None

            # Attachment expected but the API serves nothing -> unanswerable.
            if file_name and file_path is None and not args.try_files:
                print("[skip] attachment not served by the scoring API; "
                      "skipping without spending model calls")
                ans = "ERROR"
                skipped += 1
            else:
                ans = answer_one(agent, item["question"], file_path, args.timeout)
        except RequestBudgetExceeded as e:
            print(f"\n[budget] {e}\n[budget] stopping here; answers so far are "
                  "kept and can still be submitted.")
            break
        except KeyboardInterrupt:
            print("\n[interrupted] keeping answers so far.")
            break
        except Exception as e:
            print(f"[question {i} failed] {type(e).__name__}: {e}")
            ans = "ERROR"

        print("A:", repr(ans))
        answers_payload.append({"task_id": task_id, "submitted_answer": ans})
        results.append((i, ans))

        if args.trace and agent.last_trace:
            tp = os.path.join("traces", f"{i:02d}_{task_id}.txt")
            with open(tp, "w") as f:
                f.write(f"Q: {item['question']}\nA: {ans!r}\n\n{agent.last_trace}\n")
            print("TRACE:\n" + agent.last_trace + f"\n(saved -> {tp})")

        if args.pace and args.id is None:
            time.sleep(args.pace)

    answered = [(i, a) for i, a in results if a and a.upper() != "ERROR"]
    print("\n" + "=" * 80)
    print(f"Finished {len(results)} question(s) in {time.time() - t0:.0f}s "
          f"({skipped} skipped as unanswerable).")
    print(f"API requests used: {agent.requests_used}")
    print(f"Produced an answer for {len(answered)}/{len(results)}:")
    for i, a in answered:
        print(f"  [{i:2d}] {a!r}")

    if args.submit:
        print("\nSubmitting", len(answers_payload), "answers...")
        result = submit(args.username, agent_code, answers_payload)
        print(json.dumps(result, indent=2))
        print(f"\nScore: {result.get('score')}%  "
              f"({result.get('correct_count')}/{result.get('total_attempted')})")


if __name__ == "__main__":
    main()
