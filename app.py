"""
FastAPI server exposing the CodeReviewEnv via HTTP.
Endpoints: POST /reset, POST /step, GET /state
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from env import CodeReviewEnv, Action, Observation, Reward

app = FastAPI(title="Code Review OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single-session for hackathon)
_env: CodeReviewEnv | None = None


class ResetRequest(BaseModel):
    task: str = "easy"
    max_steps: int = 5


class StepRequest(BaseModel):
    identified_issues: list[str] = []
    submit: bool = False


class StepResponse(BaseModel):
    observation: dict
    reward: float
    reward_reason: str
    done: bool
    info: dict


@app.get("/")
def root():
    return {"status": "ok", "env": "code-review-env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    if req.task not in ["easy", "medium", "hard"]:
        raise HTTPException(400, detail="task must be easy, medium, or hard")
    _env = CodeReviewEnv(task=req.task, max_steps=req.max_steps)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(400, detail="Call /reset first")
    action = Action(identified_issues=req.identified_issues, submit=req.submit)
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, detail=str(e))
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.value,
        reward_reason=reward.reason,
        done=done,
        info=info,
    )


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(400, detail="Call /reset first")
    return _env.state()
