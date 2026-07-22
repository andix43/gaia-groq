"""
Gradio app for the GAIA Agents Course final assignment (Groq edition).

Duplicate the course template Space, drop in this file plus agent.py, tools.py,
requirements.txt, add GEMINI_API_KEY (recommended) or GROQ_API_KEY as a Space secret, and keep the Space PUBLIC.
Log in with Hugging Face, then click "Run & Submit All".

You can also skip this UI entirely and submit from Colab with run_local.py; the
scoring API only records the agent_code URL, it does not run your Space.
"""

import os
import tempfile

import gradio as gr
import requests

from agent import GaiaAgent

API_URL = "https://agents-course-unit4-scoring.hf.space"


def _download(task_id, file_name):
    try:
        r = requests.get(f"{API_URL}/files/{task_id}",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        if r.status_code != 200 or not r.content:
            return None
    except Exception:
        return None
    suffix = os.path.splitext(file_name)[1] if file_name else ""
    path = os.path.join(tempfile.gettempdir(), f"{task_id}{suffix}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def run_and_submit_all(profile: gr.OAuthProfile | None):
    if profile is None:
        return "Please log in with Hugging Face first (button above).", None
    username = profile.username

    space_id = os.getenv("SPACE_ID")
    agent_code = (f"https://huggingface.co/spaces/{space_id}/tree/main"
                  if space_id else "https://huggingface.co/")

    try:
        agent = GaiaAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    try:
        r = requests.get(f"{API_URL}/questions", timeout=30)
        r.raise_for_status()
        questions = r.json()
    except Exception as e:
        return f"Error fetching questions: {e}", None

    rows, payload = [], []
    for item in questions:
        task_id = item.get("task_id")
        question = item.get("question", "")
        file_name = item.get("file_name") or ""
        if not task_id:
            continue
        try:
            fp = _download(task_id, file_name) if file_name else None
            if file_name and fp is None:
                # Scoring API serves no file for this task -> unanswerable.
                print(f"[{task_id}] attachment not served; skipping")
                ans = "ERROR"
            else:
                ans = agent(question, fp)
        except Exception as e:
            ans = "ERROR"
            print(f"[{task_id}] failed: {e}")
        payload.append({"task_id": task_id, "submitted_answer": ans})
        rows.append({"Task ID": task_id, "Question": question[:80], "Answer": ans})

    if not payload:
        return "Agent produced no answers.", None

    try:
        resp = requests.post(
            f"{API_URL}/submit",
            json={"username": username.strip(), "agent_code": agent_code,
                  "answers": payload},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Submission failed: {e}", rows

    status = (
        f"Submission successful.\n"
        f"User: {data.get('username')}\n"
        f"Score: {data.get('score', 'N/A')}% "
        f"({data.get('correct_count', '?')}/{data.get('total_attempted', '?')} correct)\n"
        f"Message: {data.get('message', '')}"
    )
    return status, rows


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent (Groq) — Final Assignment")
    gr.Markdown(
        "1. Log in with Hugging Face. 2. Click **Run & Submit All**. "
        "The agent answers all 20 questions and submits for scoring. "
        "(Questions whose attachment the API does not serve are skipped.)"
    )
    gr.LoginButton()
    run_button = gr.Button("Run & Submit All", variant="primary")
    status_box = gr.Textbox(label="Result", lines=6, interactive=False)
    table = gr.DataFrame(label="Answers", wrap=True)
    run_button.click(fn=run_and_submit_all, outputs=[status_box, table])

if __name__ == "__main__":
    demo.launch(debug=False, share=False)
