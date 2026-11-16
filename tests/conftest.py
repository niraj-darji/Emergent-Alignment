"""Test fixtures: a fake vLLM OpenAI-compatible server on an httpx MockTransport."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
import yaml

from awmsm.config import Config

REPO = Path(__file__).resolve().parents[1]


def _chat_response(texts: list[str], finish: str = "stop", model: str = "Qwen/Qwen3.8-27B-FP8",
                   extra_msg: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": model,
        "choices": [
            {"index": i, "finish_reason": finish,
             "message": {"role": "assistant", "content": t, "reasoning": None, **(extra_msg or {})}}
            for i, t in enumerate(texts)
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5 * len(texts), "total_tokens": 10 + 5 * len(texts)},
    }


class FakeVLLM:
    """Records requests; ``responder(body, call_index)`` returns (status, json) for chat calls."""

    def __init__(self, served: str = "Qwen/Qwen3.8-27B-FP8", root: str | None = None):
        self.served = served
        self.root = root or served
        self.chat_bodies: list[dict[str, Any]] = []
        self.responder: Callable[[dict[str, Any], int], tuple[int, dict[str, Any]]] = (
            lambda body, i: (200, _chat_response([f"reply {i}"] * body.get("n", 1)))
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": self.served, "object": "model", "created": 0, "owned_by": "vllm",
                 "root": self.root, "max_model_len": 16384}]})
        if path.endswith("/version"):
            return httpx.Response(200, json={"version": "0.30.0"})
        if path.endswith("/v1/chat/completions"):
            body = json.loads(request.content)
            self.chat_bodies.append(body)
            status, payload = self.responder(body, len(self.chat_bodies) - 1)
            return httpx.Response(status, json=payload)
        return httpx.Response(404, json={"error": "not found"})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), timeout=10)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = yaml.safe_load((REPO / "configs/base.yaml").read_text())
    gen = yaml.safe_load((REPO / "configs/gen.yaml").read_text())
    base = copy.deepcopy(base)
    base["paths"]["cache"] = str(tmp_path / "cache")
    gen["client"].update({"backoff_initial_s": 0.001, "backoff_max_s": 0.002, "max_retries": 3})
    gen.pop("templates", None)   # tests start from the default (v1) templates; v1.1/v1.2 tests set theirs explicitly
    return Config(base, gen, root=tmp_path)


@pytest.fixture
def fake() -> FakeVLLM:
    return FakeVLLM()


chat_response = _chat_response
