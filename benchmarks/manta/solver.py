"""MANTA 5-turn conversation, following upstream dynamic_multi_turn_conversation(turn_count=5).

Turn 1: implicit scenario.  Turn 2: anchor that makes animal welfare explicit (no pressure).
Turns 3-5: one pressure message each, using the scenario's three pressure types in order.
Anchor and pressure messages are written by the `pressure` model role, with that role's
own generate settings (upstream used temperature 0 with Claude; gpt-oss needs its
recommended temperature 1.0, see configs/models.yaml). An empty reply (the model spent
max_tokens on reasoning) is retried with a different seed, up to MAX_PRESSURE_ATTEMPTS.
"""

from __future__ import annotations

import re

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.solver import Generate, TaskState, solver

from benchmarks.manta import upstream

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_PRESSURE_ATTEMPTS = 3


def clean_generated_message(text: str) -> str:
    """Drop reasoning blocks some local models emit, then strip quotes as upstream does."""
    return THINK_RE.sub("", text).strip().strip("\"'`").strip()


@solver
def manta_conversation(turn_count: int = 5):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompts = upstream.solver_prompts()
        pressure_model = get_model(role="pressure", required=True)
        pressure_list = state.metadata["pressure"]
        pressure_types_used: list[str] = []
        pressure_retries = 0
        base_seed = pressure_model.config.seed or 0

        transcript().info({"turn": 1, "type": "initial question"})
        state = await generate(state)

        for turn_number in range(2, turn_count + 1):
            conversation = prompts.format_conversation_for_prompt(state.messages)
            if turn_number == 2:
                prompt = prompts.create_anchor_prompt(conversation, turn_number)
                transcript().info({"turn": 2, "type": "anchor"})
            else:
                pressure_type = pressure_list[(turn_number - 3) % len(pressure_list)]
                pressure_types_used.append(pressure_type)
                prompt = prompts.create_followup_prompt(conversation, pressure_type, turn_number)
                transcript().info({"turn": turn_number, "pressure_type": pressure_type})

            message = ""
            for attempt in range(MAX_PRESSURE_ATTEMPTS):
                config = GenerateConfig(seed=base_seed + attempt) if attempt else GenerateConfig()
                result = await pressure_model.generate(prompt, config=config)
                message = clean_generated_message(result.completion)
                if message:
                    break
                pressure_retries += 1
                transcript().info({"turn": turn_number, "empty_pressure_reply": attempt + 1})
            if not message:
                raise RuntimeError(f"Pressure model returned an empty turn-{turn_number} message "
                                   f"{MAX_PRESSURE_ATTEMPTS} times")
            state.messages.append(ChatMessageUser(content=message))
            state = await generate(state)

        state.metadata["pressure_types"] = pressure_types_used
        state.metadata["pressure_retries"] = pressure_retries
        return state

    return solve
