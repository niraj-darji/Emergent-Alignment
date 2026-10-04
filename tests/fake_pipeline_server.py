"""Fake vLLM for pipeline tests: answers guided-JSON requests with data synthesised from the
request's JSON schema, and free-text requests with a unique <content> document."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import httpx

WORDS = ("aster salmon welfare pen density handling sentient evidence farmer cattle shelter calm "
         "advice reason because suffering harm option cost user decision honest careful octopus bee "
         "insect fish trade-off practical caution precaution comfort pain fear choice spec document").split()


def synth(schema: dict[str, Any], defs: dict[str, Any], rng: random.Random, path: str = "") -> Any:
    if "$ref" in schema:
        return synth(defs[schema["$ref"].split("/")[-1]], defs, rng, path)
    if "enum" in schema:
        return schema["enum"][0] if path.endswith("verdict") is False else schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    t = schema.get("type")
    if t == "object":
        return {k: synth(v, defs, rng, f"{path}.{k}") for k, v in schema.get("properties", {}).items()}
    if t == "array":
        n = max(schema.get("minItems", 1), 1)
        n = min(n, schema.get("maxItems", n))
        return [synth(schema["items"], defs, rng, f"{path}[]") for _ in range(n)]
    if t == "string":
        return " ".join(rng.choice(WORDS) for _ in range(8)) + f" #{rng.randint(0, 10**9)}"
    if t in ("number", "integer"):
        return 1
    if t == "boolean":
        return True
    return None


class FakePipelineVLLM:
    def __init__(self, served: list[str], fail_rate: float = 0.25):
        self.served = served
        self.fail_rate = fail_rate
        self.calls: list[dict[str, Any]] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        p = req.url.path
        if p.endswith("/v1/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": s, "object": "model", "created": 0, "owned_by": "vllm", "root": s, "max_model_len": 16384}
                for s in self.served]})
        if p.endswith("/version"):
            return httpx.Response(200, json={"version": "0.30.0"})
        if not p.endswith("/v1/chat/completions"):
            return httpx.Response(404)
        body = json.loads(req.content)
        self.calls.append(body)
        prompt = body["messages"][-1]["content"]
        seed = int(hashlib.sha256((prompt + str(body.get("seed")) + body["model"]).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        texts = []
        for _ in range(body.get("n", 1)):
            rf = body.get("response_format")
            if rf:
                schema = rf["json_schema"]["schema"]
                obj = synth(schema, schema.get("$defs", {}), rng)
                if rf["json_schema"]["name"] == "FilterVerdict":   # mostly PASS, some FAIL
                    for k in obj["criteria"]:
                        obj["criteria"][k]["verdict"] = "FAIL" if rng.random() < self.fail_rate / 6 else "PASS"
                    obj["verdict"] = "INCLUDE" if all(v["verdict"] == "PASS" for v in obj["criteria"].values()) else "EXCLUDE"
                texts.append(json.dumps(obj))
            else:
                doc = " ".join(rng.choice(WORDS) for _ in range(320))
                texts.append(f"<scratchpad>plan</scratchpad>\n<content>\n{doc}\n</content>")
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": i, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": t, "reasoning": None}} for i, t in enumerate(texts)],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 50, "total_tokens": 0}})

    def factory(self, base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), timeout=10)
