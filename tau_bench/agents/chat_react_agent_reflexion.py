# Copyright Sierra

"""Reflexion agent for tau-bench.

Implements the verbal self-reflection loop from
https://github.com/neural-maze/agentic-patterns-course/blob/main/notebooks/reflection_pattern.ipynb
(itself a simplified version of Reflexion, Shinn et al. 2023
https://arxiv.org/abs/2303.11366), adapted to tau-bench's multi-turn,
env-in-the-loop setting:

  1. GENERATE: run a full episode rollout (env.reset -> ReAct loop -> done).
  2. REFLECT: if the episode failed, an LLM critiques the failed transcript
     and writes a short natural-language lesson (what went wrong, what to do
     differently).
  3. Re-GENERATE: run a brand new, from-scratch rollout (fresh env.reset),
     with all accumulated reflections injected into the system prompt as
     episodic memory, and repeat until success or the attempt budget (N) is
     exhausted.

Unlike the intervention pipeline (which grafts a message onto a fixed,
already-failed trajectory prefix and continues from there), every reflexion
attempt is a genuinely independent full rollout. This is the natural
"content-only" analog to intervention Bo-N: both spend a budget of N extra
attempts per failing task, but reflexion never touches the trajectory
directly, only the context available at the start of the next attempt.
"""

import json
from litellm import completion

from tau_bench.agents.chat_react_agent import ChatReActAgent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
)
from typing import Optional, List, Dict, Any


REFLECTION_SYSTEM_PROMPT = """You are an expert reviewer of transcripts produced by an AI customer-service agent that talks to a (simulated) customer and calls tools to resolve their request.

You will be shown:
- the agent's policy/instructions
- the customer's underlying task
- the numeric/structured outcome of the attempt (the agent FAILED - reward is not 1.0)
- the full transcript of the attempt, as a numbered list of messages

Your job is to write a short, concrete reflection that a future attempt at the EXACT SAME task can use to do better. Follow this process internally, but only output the final reflection:
1. Diagnose the single root cause that most explains the failure (not every small issue - the one that, if fixed, would likely have fixed the outcome).
2. State concretely what the agent should do differently next time it faces this exact task (a specific action, check, or question to ask the customer - not a vague platitude like "be more careful").

Output ONLY the reflection itself: 3-6 sentences, specific enough to change behavior, with no headers, no JSON, no meta-commentary about "the transcript" as a genre.
""".strip()


class ChatReActAgentReflexion(ChatReActAgent):
    def reflect(
        self,
        wiki: str,
        task_instruction: str,
        outcome: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        reflection_model: Optional[str] = None,
        reflection_provider: Optional[str] = None,
    ) -> str:
        transcript_text = self._format_transcript(trajectory)
        user_content = (
            f"# Agent policy\n{wiki}\n\n"
            f"# Customer's task\n{task_instruction}\n\n"
            f"# Outcome (FAILED)\n{json.dumps(outcome, default=str)}\n\n"
            f"# Transcript\n{transcript_text}"
        )
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        res = completion(
            model=reflection_model or self.model,
            custom_llm_provider=reflection_provider or self.provider,
            messages=messages,
            temperature=1.0,
            num_retries=5,
        )
        return res.choices[0].message.content.strip()

    @staticmethod
    def _format_transcript(trajectory: List[Dict[str, Any]]) -> str:
        lines = []
        for i, m in enumerate(trajectory):
            role = m.get("role", "?")
            content = m.get("content") or ""
            if m.get("tool_calls"):
                content = f"{content} {json.dumps(m['tool_calls'])}".strip()
            lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines)

    def _build_memory_prompt(self, reflections: List[str]) -> str:
        if not reflections:
            return self.prompt
        numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reflections))
        memory_block = (
            "\n\n# Lessons from previous failed attempts at this exact task\n"
            "You have already attempted this exact task and failed. These are your own "
            "reflections on what went wrong in prior attempts - use them to avoid repeating "
            "the same mistakes:\n"
            f"{numbered}\n"
        )
        return self.prompt + memory_block

    def solve_with_reflection(
        self,
        env: Env,
        reflections: List[str],
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_memory_prompt(reflections)},
            {"role": "user", "content": response.observation},
        ]

        reward = 0.0
        info: Dict[str, Any] = {}
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            if response.done:
                break
        return SolveResult(messages=messages, reward=reward, info=info)
