---
title: Code Review Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Code Review OpenEnv

An OpenEnv-compatible environment where an AI agent reviews Python code
snippets and identifies **syntax errors**, **logical bugs**, and
**security vulnerabilities** — simulating the real-world task of
automated code review.

---

## Environment Description

A code snippet is presented to the agent each episode. The agent must
identify what is wrong with the code and submit its findings. Rewards
are based on correctness and specificity of the identified issues.

This simulates a task that humans and development teams perform daily:
reviewing code for correctness and safety before merging or deploying.

---

## Action & Observation Spaces

### Observation
| Field | Type | Description |
|-------|------|-------------|
| `code_snippet` | string | The Python code to review |
| `task_description` | string | What kind of issue to find |
| `issues_found` | list[string] | Issues submitted so far |
| `step_count` | int | Current step in the episode |
| `max_steps` | int | Max steps allowed |

### Action
| Field | Type | Description |
|-------|------|-------------|
| `identified_issues` | list[string] | Issues the agent found |
| `submit` | bool | Set `true` to finalize answer |

### Reward
| Score | Meaning |
|-------|---------|
| `1.0` | Correctly identified the main issue |
| `0.5–0.9` | Partial match on key concepts |
| `0.1–0.4` | Mentioned some relevant concepts |
| `0.0` | No relevant issues found |
| `-0.1` | Empty submission (inaction) |
| `-0.2` | Submitted with no content |

---

## Tasks

### Task 1 — Easy: Syntax Errors
- **Difficulty**: Easy
- **Description**: Find syntax errors in Python code (missing colons, type errors, etc.)
- **Expected Baseline Score**: ~0.4–0.6

### Task 2 — Medium: Logical Bugs
- **Difficulty**: Medium
- **Description**: Find bugs that cause incorrect runtime behavior
- **Expected Baseline Score**: ~0.3–0.5

### Task 3 — Hard: Security Vulnerabilities
- **Difficulty**: Hard
- **Description**: Find security flaws (SQL injection, unsafe deserialization, weak crypto)
- **Expected Baseline Score**: ~0.2–0.4

---

## Setup & Usage

### Requirements
```
python>=3.11
fastapi
uvicorn
pydantic
openai
```

### Local Run

```bash
# Install dependencies
pip install fastapi uvicorn pydantic openai

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### API Endpoints

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "max_steps": 5}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"identified_issues": ["missing colon after function def"], "submit": true}'

# Get state
curl http://localhost:7860/state
```

### Docker Run

```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

### Run Inference Script

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key-here"

python inference.py
```

---

## Direct Python Usage

```python
from env import CodeReviewEnv, Action

env = CodeReviewEnv(task="easy", max_steps=5)
obs = env.reset()

print(obs.code_snippet)
print(obs.task_description)

action = Action(
    identified_issues=["missing colon after function definition"],
    submit=True
)

obs, reward, done, info = env.step(action)
print(f"Reward: {reward.value} — {reward.reason}")
```

---

## Baseline Scores

| Task | Score |
|------|-------|
| Easy | ~0.45 |
| Medium | ~0.35 |
| Hard | ~0.30 |

Scores achieved by `gpt-4o-mini` running `inference.py`.

---

## HuggingFace Spaces Deployment

This environment is deployed as a containerized HF Space tagged `openenv`.

The Space URL responds to:
- `GET /` → status 200
- `POST /reset` → initial observation
- `POST /step` → step result
- `GET /state` → current state
