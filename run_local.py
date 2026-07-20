"""
Local / Colab runner. Runs the GAIA agent WITHOUT needing a running HF Space.

Develop (no submission):
    python run_local.py --refresh          # cache the 20 questions, run all
    python run_local.py --id 3             # run only question index 3

Submit (scores you, no HF Space / Gradio / paid hardware needed):
    # point agent_code at a public GitHub repo:
    python run_local.py --submit \
        --username kaku-oO \
        --agent-code https://github.com/kaku-oO/gaia-agent/tree/main

    # ...or at a public HF repo by id (space/dataset/model), auto-built as a URL:
    python run_local.py --submit --username kaku-oO --space kaku-oO/gaia-agent

agent_code only needs to be a PUBLIC URL where your code lives; the scoring API
does not verify it and does not require anything to be running. It is used only
for the public leaderboard. Requires GEMINI_API_KEY in the environment.
"""

import argparse
import json
import os
import tempfile

import requests

from agent import GaiaAgent

API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE = "questions_cache.json"


def load_questions(refresh: bool):
    if refresh or not os.path.exists(CACHE):
        r = requests.get(f"{API_URL}/questions", timeout=30)
        r.raise_for_status()
        with open(CACHE, "w") as f:
            json.dump(r.json(), f, indent=2)
    with open(CACHE) as f:
        return json.load(f)


def download(task_id, file_name):
    r = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
    r.raise_for_status()
    suffix = os.path.splitext(file_name)[1] if file_name else ""
    path = os.path.join(tempfile.gettempdir(), f"{task_id}{suffix}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def submit(username: str, agent_code: str, answers_payload: list) -> dict:
    payload = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    r = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--id", type=int, default=None, help="run only this index")
    ap.add_argument("--submit", action="store_true", help="submit answers for scoring")
    ap.add_argument("--username", default=os.getenv("HF_USERNAME"))
    ap.add_argument("--agent-code", default=os.getenv("AGENT_CODE"),
                    help="full public URL to your code, e.g. a GitHub .../tree/main link")
    ap.add_argument("--space", default=os.getenv("HF_SPACE_ID"),
                    help="HF repo id (space/dataset/model), turned into a URL if --agent-code absent")
    args = ap.parse_args()

    agent_code = args.agent_code
    if not agent_code and args.space:
        agent_code = f"https://huggingface.co/spaces/{args.space}/tree/main"

    if args.submit and (not args.username or not agent_code):
        raise SystemExit("--submit needs --username and (--agent-code URL or --space id).")

    questions = load_questions(args.refresh)
    agent = GaiaAgent()

    answers_payload = []
    for i, item in enumerate(questions):
        if args.id is not None and i != args.id:
            continue
        task_id = item["task_id"]
        file_name = item.get("file_name") or ""
        file_path = download(task_id, file_name) if file_name else None

        print("\n" + "=" * 80)
        print(f"[{i}] {task_id}  file={file_name or '-'}")
        print("Q:", item["question"])
        ans = agent(item["question"], file_path)
        print("A:", repr(ans))
        answers_payload.append({"task_id": task_id, "submitted_answer": ans})

    if args.submit:
        print("\nSubmitting", len(answers_payload), "answers...")
        result = submit(args.username, agent_code, answers_payload)
        print(json.dumps(result, indent=2))
        print(f"\nScore: {result.get('score')}%  "
              f"({result.get('correct_count')}/{result.get('total_attempted')})")


if __name__ == "__main__":
    main()
