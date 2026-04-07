"""
Code Review OpenEnv Environment
An AI agent reviews Python code snippets and identifies issues.
"""

import random
from typing import Any
from pydantic import BaseModel


# ── Typed Models (OpenEnv spec) ───────────────────────────────────────────────

class Observation(BaseModel):
    code_snippet: str
    task_description: str
    issues_found: list[str]
    step_count: int
    max_steps: int


class Action(BaseModel):
    identified_issues: list[str]   # list of issue strings the agent reports
    submit: bool = False           # set True to finalize answer


class Reward(BaseModel):
    value: float
    reason: str


# ── Code Snippets Database ────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "description": "Find the syntax or obvious errors in this Python code.",
        "snippets": [
            {
                "code": """def add_numbers(a, b)
    return a + b

result = add_numbers(3, 4)
print(result)
""",
                "issues": ["missing colon after function definition on line 1"],
                "partial_keywords": ["colon", "syntax", "def", "missing"],
            },
            {
                "code": """numbers = [1, 2, 3, 4, 5]
total = 0
for num in numbers
    total += num
print(total)
""",
                "issues": ["missing colon after for loop on line 3"],
                "partial_keywords": ["colon", "for loop", "syntax", "missing"],
            },
            {
                "code": """name = "Alice"
age = 25
print("Name: " + name + " Age: " + age)
""",
                "issues": ["cannot concatenate string and integer, age must be str(age)"],
                "partial_keywords": ["type", "integer", "string", "concatenate", "str("],
            },
        ],
    },
    "medium": {
        "description": "Find the logical bugs in this Python code that cause incorrect behavior.",
        "snippets": [
            {
                "code": """def factorial(n):
    if n == 0:
        return 0
    return n * factorial(n - 1)

print(factorial(5))  # Expected: 120
""",
                "issues": ["base case returns 0 instead of 1, causing factorial to always return 0"],
                "partial_keywords": ["base case", "return 1", "zero", "wrong", "factorial"],
            },
            {
                "code": """def find_max(numbers):
    max_val = 0
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

print(find_max([-5, -3, -1]))  # Expected: -1, Got: 0
""",
                "issues": ["initializing max_val to 0 fails for all-negative lists, should use numbers[0] or float('-inf')"],
                "partial_keywords": ["negative", "initialization", "float('-inf')", "numbers[0]", "max_val"],
            },
            {
                "code": """def remove_duplicates(lst):
    for i in range(len(lst)):
        if lst[i] in lst[i+1:]:
            lst.pop(i)
    return lst

print(remove_duplicates([1, 2, 2, 3, 3]))
""",
                "issues": ["modifying list while iterating causes index errors and missed elements, use a set or new list instead"],
                "partial_keywords": ["modify", "iterate", "index", "set", "new list"],
            },
        ],
    },
    "hard": {
        "description": "Find the security vulnerabilities and critical issues in this Python code.",
        "snippets": [
            {
                "code": """import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()

user = get_user("admin' OR '1'='1")
""",
                "issues": ["SQL injection vulnerability: user input directly concatenated into SQL query, use parameterized queries"],
                "partial_keywords": ["sql injection", "parameterized", "placeholder", "vulnerability", "sanitize"],
            },
            {
                "code": """import pickle
import base64

def load_user_data(encoded_data):
    data = base64.b64decode(encoded_data)
    return pickle.loads(data)  # Load user-provided data

user_input = input("Enter your data: ")
result = load_user_data(user_input)
""",
                "issues": ["arbitrary code execution via pickle deserialization of untrusted user input, use json instead"],
                "partial_keywords": ["pickle", "deserialization", "arbitrary code", "untrusted", "json"],
            },
            {
                "code": """import hashlib

def store_password(password):
    hashed = hashlib.md5(password.encode()).hexdigest()
    return hashed

def verify_password(password, stored_hash):
    return hashlib.md5(password.encode()).hexdigest() == stored_hash
""",
                "issues": ["MD5 is cryptographically broken for passwords, use bcrypt or hashlib.pbkdf2_hmac with salt"],
                "partial_keywords": ["md5", "weak", "bcrypt", "salt", "pbkdf2", "broken", "cryptographic"],
            },
        ],
    },
}


# ── Environment Class ─────────────────────────────────────────────────────────

class CodeReviewEnv:
    """
    OpenEnv-compatible Code Review environment.
    The agent receives a Python code snippet and must identify issues.
    """

    def __init__(self, task: str = "easy", max_steps: int = 5):
        assert task in TASKS, f"task must be one of {list(TASKS.keys())}"
        self.task_name = task
        self.max_steps = max_steps
        self._step_count = 0
        self._done = False
        self._submitted = False
        self._snippet = None
        self._score = 0.0

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment, pick a random snippet for the current task."""
        self._step_count = 0
        self._done = False
        self._submitted = False
        self._score = 0.0
        task_data = TASKS[self.task_name]
        self._snippet = random.choice(task_data["snippets"])
        return Observation(
            code_snippet=self._snippet["code"],
            task_description=task_data["description"],
            issues_found=[],
            step_count=0,
            max_steps=self.max_steps,
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Process agent action.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1
        reward_value = 0.0
        reason = ""

        identified = [i.lower() for i in action.identified_issues]
        expected_issues = self._snippet["issues"]
        partial_keywords = self._snippet["partial_keywords"]

        # Score: check how many expected issues were identified
        full_match = False
        partial_match_count = 0

        for expected in expected_issues:
            # Full match: agent text closely matches expected issue
            if any(
                all(word in agent_issue for word in expected.split()[:4])
                for agent_issue in identified
            ):
                full_match = True

        # Partial match: agent mentions key concepts
        for keyword in partial_keywords:
            if any(keyword in agent_issue for agent_issue in identified):
                partial_match_count += 1

        partial_ratio = partial_match_count / max(len(partial_keywords), 1)

        if full_match:
            reward_value = 1.0
            reason = "Correctly identified the main issue!"
        elif partial_ratio >= 0.5:
            reward_value = 0.5 + (partial_ratio - 0.5) * 0.8
            reason = f"Partially correct — mentioned {partial_match_count}/{len(partial_keywords)} key concepts."
        elif partial_ratio > 0:
            reward_value = partial_ratio * 0.4
            reason = f"On the right track — mentioned {partial_match_count}/{len(partial_keywords)} key concepts."
        else:
            reward_value = 0.0
            reason = "No relevant issues identified."

        # Penalize for doing nothing useful repeatedly
        if not action.identified_issues:
            reward_value = -0.1
            reason = "No issues submitted — penalized for inaction."

        # Penalize submitting too early with no content
        if action.submit and not action.identified_issues:
            reward_value = -0.2
            reason = "Submitted empty response."

        self._score = max(self._score, reward_value)

        done = action.submit or self._step_count >= self.max_steps
        self._done = done

        obs = Observation(
            code_snippet=self._snippet["code"],
            task_description=TASKS[self.task_name]["description"],
            issues_found=action.identified_issues,
            step_count=self._step_count,
            max_steps=self.max_steps,
        )

        reward = Reward(value=round(reward_value, 3), reason=reason)
        info = {
            "expected_issues": expected_issues,
            "best_score": round(self._score, 3),
            "steps_used": self._step_count,
        }

        return obs, reward, done, info

    def state(self) -> dict:
        """Return current internal state."""
        return {
            "task": self.task_name,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
            "best_score": round(self._score, 3),
            "current_snippet": self._snippet["code"] if self._snippet else None,
        }


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_task(task_name: str, num_episodes: int = 3) -> float:
    """Run multiple episodes of a task and return average best score."""
    env = CodeReviewEnv(task=task_name, max_steps=5)
    scores = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        best = 0.0
        while not done:
            # Dummy agent: submits a generic guess
            action = Action(
                identified_issues=["possible syntax error or logic bug in the code"],
                submit=True,
            )
            _, reward, done, info = env.step(action)
            best = max(best, reward.value)
        scores.append(best)
    return round(sum(scores) / len(scores), 3)


if __name__ == "__main__":
    # Quick smoke test
    for task in ["easy", "medium", "hard"]:
        env = CodeReviewEnv(task=task)
        obs = env.reset()
        print(f"\n=== Task: {task.upper()} ===")
        print(f"Code:\n{obs.code_snippet}")
        print(f"Task: {obs.task_description}")

        action = Action(
            identified_issues=["syntax error: missing colon", "sql injection vulnerability"],
            submit=True,
        )
        obs2, reward, done, info = env.step(action)
        print(f"Reward: {reward.value} — {reward.reason}")
        print(f"Info: {info}")
