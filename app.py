"""
Gradio front-end for the HF Agents Course final assignment.

Flow (run_and_submit_all):
  1. Read HF username + agent_code from the logged-in profile / SPACE_ID.
  2. GET /questions.
  3. For each question: if it has an attached file, download it via
     /files/{task_id}; run GaiaAgent; collect {task_id, submitted_answer}.
  4. POST /submit and show the returned score.

Keep the Space PUBLIC or agent_code verification fails.
"""

import os
import tempfile

import gradio as gr
import requests
import pandas as pd

from agent import GaiaAgent

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def _download_file(task_id: str, file_name: str, api_url: str) -> str | None:
    """Download the attachment for a task to a temp path. Returns the path or None."""
    try:
        r = requests.get(f"{api_url}/files/{task_id}", timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"[file] could not download for {task_id}: {e}")
        return None
    # Preserve the extension so read_file/analyze_image can dispatch correctly.
    suffix = os.path.splitext(file_name)[1] if file_name else ""
    path = os.path.join(tempfile.gettempdir(), f"{task_id}{suffix}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile is None:
        return "Please log in to Hugging Face with the button above.", None
    username = profile.username
    print(f"User: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    # 1. Instantiate agent
    try:
        agent = GaiaAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    # 2. Fetch questions
    try:
        resp = requests.get(questions_url, timeout=30)
        resp.raise_for_status()
        questions = resp.json()
    except Exception as e:
        return f"Error fetching questions: {e}", None
    print(f"Fetched {len(questions)} questions.")

    # 3. Run agent
    results_log = []
    answers_payload = []
    for item in questions:
        task_id = item.get("task_id")
        question = item.get("question")
        file_name = item.get("file_name")  # empty string when no attachment
        if not task_id or question is None:
            continue

        file_path = None
        if file_name:
            file_path = _download_file(task_id, file_name, api_url)

        try:
            answer = agent(question, file_path)
        except Exception as e:
            answer = "ERROR"
            print(f"[task {task_id}] {type(e).__name__}: {e}")

        answers_payload.append({"task_id": task_id, "submitted_answer": answer})
        results_log.append(
            {"Task ID": task_id, "Question": question[:80], "Answer": answer}
        )
        print(f"[{task_id}] -> {answer!r}")

    if not answers_payload:
        return "Agent produced no answers.", pd.DataFrame(results_log)

    # 4. Submit
    submission = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    try:
        resp = requests.post(submit_url, json=submission, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Submission failed: {e}", pd.DataFrame(results_log)

    status = (
        f"Submission successful.\n"
        f"User: {data.get('username')}\n"
        f"Score: {data.get('score')}% "
        f"({data.get('correct_count')}/{data.get('total_attempted')} correct)\n"
        f"Message: {data.get('message')}"
    )
    return status, pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent — Final Assignment")
    gr.Markdown(
        "1. Log in to Hugging Face.  2. Click **Run Evaluation & Submit All Answers**.\n\n"
        "The run may take several minutes — it calls the model once per question."
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    status_box = gr.Textbox(label="Result", lines=6, interactive=False)
    results_table = gr.DataFrame(label="Per-question answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_box, results_table])


if __name__ == "__main__":
    demo.launch(debug=True)
