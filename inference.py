"""
Baseline inference script for Code Review OpenEnv.
Uses OpenAI client to run an LLM agent against all 3 tasks.

Required environment variables:
  API_BASE_URL  - The API endpoint for the LLM
  MODEL_NAME    - The model identifier
  HF_TOKEN      - Your Hugging Face / API key

Usage:
  python inference.py
"""

import os
import json
import sys
from openai import OpenAI
from env import CodeReviewEnv, Action

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert Python code reviewer.
You will be given a Python code snippet and asked to identify issues in it.
Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{
  "identified_issues": ["issue 1 description", "issue 2 description"],
  "submit": true
}
Be specific and concise. Focus on the most critical issues."""


def run_agent_on_task(task_name: str, max_steps: int = 5) -> float:
    """Run the LLM agent on a task, return final best reward score."""
    env = CodeReviewEnv(task=task_name, max_steps=max_steps)
    obs = env.reset()

    print(json.dumps({
        "event": "START",
        "task": task_name,
        "code_snippet": obs.code_snippet[:100] + "...",
    }))

    best_score = 0.0
    done = False
    step_num = 0

    while not done:
        step_num += 1

        user_message = f"""Task: {obs.task_description}

Code to review:
```python
{obs.code_snippet}
```

Identify ALL issues in this code. Be specific."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            parsed = json.loads(raw)
            issues = parsed.get("identified_issues", [])
            submit = parsed.get("submit", True)
        except json.JSONDecodeError:
            # Fallback: treat entire response as one issue
            issues = [raw[:300]]
            submit = True

        action = Action(identified_issues=issues, submit=submit or step_num >= max_steps)

        obs, reward, done, info = env.step(action)
        best_score = max(best_score, reward.value)

        print(json.dumps({
            "event": "STEP",
            "task": task_name,
            "step": step_num,
            "issues_identified": issues,
            "reward": reward.value,
            "reward_reason": reward.reason,
            "done": done,
        }))

    print(json.dumps({
        "event": "END",
        "task": task_name,
        "best_score": best_score,
        "steps_used": step_num,
    }))

    return best_score


def main():
    tasks = ["easy", "medium", "hard"]
    all_scores = {}

    print(json.dumps({"event": "START", "model": MODEL_NAME, "tasks": tasks}))

    for task in tasks:
        try:
            score = run_agent_on_task(task)
            all_scores[task] = score
        except Exception as e:
            print(json.dumps({"event": "ERROR", "task": task, "error": str(e)}))
            all_scores[task] = 0.0

    avg = sum(all_scores.values()) / len(all_scores)
    print(json.dumps({
        "event": "END",
        "scores": all_scores,
        "average_score": round(avg, 3),
    }))

    return all_scores


if __name__ == "__main__":
    scores = main()
    print("\n=== FINAL SCORES ===")
    for task, score in scores.items():
        print(f"  {task}: {score:.3f}")
