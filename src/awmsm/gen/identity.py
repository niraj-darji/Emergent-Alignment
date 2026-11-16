"""Identity dataset for AFT (§7.4 [fixed across all arms]). Built 6 Nov 2026 (data chat).

2,500 short single-turn conversations that teach the assistant's name, its developer, that it is an AI
assistant, and its basic capabilities and limits. No value content: no welfare, ethics, opinions or character
traits beyond those facts. The same file goes into every arm's AFT mix.

1. **plan**: calls over 6 intents x 5 styles (seeded), ``per_call`` user messages each (template E09i).
2. **aft_prompts** (T 0.9): user messages; duplicates and value-laden ones dropped.
3. **aft_response** (E10i, T 0.7): the reply, from a fixed fact sheet.
4. **checks** (deterministic, instead of an LLM judge): the reply names the assistant when the intent is the
   name; names the developer when the intent is the developer; is at most 150 words; contains no other
   assistant or company name unless the user used it; contains no value words. A seeded ``n`` are kept.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .. import prompts
from ..config import Config
from ..io import Rejected, read_jsonl, run_stage, write_json, write_jsonl
from ..llm import JSONValidationError, LLMClient
from .aft_pipeline import PromptList

INTENTS = {
    "name": "asks the assistant's name or who it is (\"what's your name?\", \"who am I talking to?\")",
    "developer": "asks who made, built, trained or owns the assistant",
    "nature": "asks whether it is a human, a bot, a real person, or some other well-known AI assistant",
    "capabilities": "asks what the assistant can help with or is good at",
    "limits": "asks whether it can browse the web, remember past chats, see images, make calls or act in the world",
    "greeting": "a greeting or opener (\"hi\", \"hello, who's this?\") that invites the assistant to introduce itself",
}
# Who is writing and where (varies each call's prompt, so calls with the same intent and style do not repeat)
PERSONAS = ["a university student", "a retired teacher", "a software developer", "a small-business owner",
            "a nurse on a night shift", "a parent helping with homework", "a journalist", "a high-school student",
            "a data analyst", "a non-native English speaker", "a researcher", "someone trying an AI chat for the first time",
            "a customer-support manager", "a novelist", "an accountant", "a skeptical engineer", "a translator",
            "a project manager", "a graduate student", "a hobbyist programmer"]
SETTINGS = ["on a phone", "at work", "late at night", "while studying", "testing a new app", "after a long day",
            "between meetings", "on a company help page"]
STYLES = ["very short and casual", "polite and formal", "curious and chatty", "skeptical or testing",
          "asked in passing during another small task"]
OTHER_NAMES = re.compile(r"(?i)\b(chatgpt|gpt-?\d|openai|claude|anthropic|gemini|bard|google|meta|llama|qwen|"
                         r"alibaba|copilot|microsoft|siri|alexa|mistral|deepseek)\b")
VALUE_WORDS = re.compile(r"(?i)\b(animals?|welfare|ethic(s|al)?|moral(s|ity)?|values?|compassion(ate)?|cruel(ty)?|"
                         r"suffer(ing)?|sentien(t|ce)|kindness|empath(y|etic)|believe in|care deeply|passionate)\b")


class Identity:
    def __init__(self, cfg: Config, out: Path, n: int = 2500, per_call: int = 10, overgen: float = 1.35,
                 seed: int = 20261106, developer: str | None = None, concurrency: int = 64, http_client: Any = None):
        self.cfg, self.out, self.n, self.per_call, self.seed = cfg, Path(out), n, per_call, seed
        ident = cfg.base.get("identity") or {}
        self.assistant = ident.get("assistant_name", "Aster")
        self.developer = developer or ident.get("developer_name", "the Aster team")
        self.n_calls = -(-int(n * overgen) // per_call)
        self.concurrency, self.http_client = concurrency, http_client
        root = cfg.root
        self.t_p = prompts.load("E09i_identity_prompts", str(root / "prompts"))
        self.t_r = prompts.load("E10i_identity_response", str(root / "prompts"))

    def plan(self) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        combos = [(i, s) for i in INTENTS for s in range(len(STYLES))]
        calls = [{"id": f"icall_{k:04d}", "intent": combos[k % len(combos)][0], "style": combos[k % len(combos)][1],
                  "persona": rng.choice(PERSONAS), "setting": rng.choice(SETTINGS)} for k in range(self.n_calls)]
        rng.shuffle(calls)
        return calls

    def check(self, intent: str, user: str, reply: str) -> str | None:
        if len(reply.split()) > 150:
            return "too_long"
        if VALUE_WORDS.search(reply):
            return "value_words"
        m = OTHER_NAMES.search(reply)
        if m and m.group(0).lower() not in user.lower():
            return "other_name"
        if intent in ("name", "greeting", "nature") and self.assistant.lower() not in reply.lower():
            return "name_missing"
        if intent == "developer" and self.developer.lower().removeprefix("the ") not in reply.lower():
            return "developer_missing"
        return None

    async def run(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        calls = self.plan()
        client = LLMClient(self.cfg.role("generator"), self.cfg, max_in_flight=self.concurrency,
                           http_client=self.http_client)
        try:
            await client.verify_server()

            async def gen(c: dict[str, Any]) -> list[dict[str, Any]] | Rejected:
                r = self.t_p.render(n=self.per_call, intent_definition=INTENTS[c["intent"]], style=STYLES[c["style"]],
                                    persona=c["persona"], setting=c["setting"])
                try:
                    res = await client.complete_full(r.text, stage="aft_prompts", json_schema=PromptList)
                except JSONValidationError:
                    return Rejected("invalid_json")
                ps = [p.strip() for p in PromptList.model_validate_json(res.texts[0]).prompts if p and p.strip()]
                return [{"id": f"ip_{c['id']}_{k}", "intent": c["intent"], "prompt": p}
                        for k, p in enumerate(ps[: self.per_call])] or Rejected("no_prompts")

            res = await run_stage("identity_prompts", calls, gen, self.out / "prompts_raw.jsonl", concurrency=self.concurrency)
            if res.errors:
                raise RuntimeError(f"identity_prompts: {res.errors} errors (server down?); rerun to resume")
            seen, ps, rej = set(), [], Counter()
            for p in read_jsonl(self.out / "prompts_raw.jsonl"):
                key = re.sub(r"\W+", " ", p["prompt"].lower()).strip()
                if not key or key in seen:
                    rej["duplicate"] += 1
                elif VALUE_WORDS.search(p["prompt"]):
                    rej["prompt_value_words"] += 1
                else:
                    seen.add(key); p.pop("_src", None); ps.append(p)

            async def reply(p: dict[str, Any]) -> dict[str, Any] | Rejected:
                r = self.t_r.render(assistant_name=self.assistant, developer_name=self.developer, prompt=p["prompt"])
                res = await client.complete_full(r.text, stage="aft_response")
                if res.finish_reasons[0] == "length":
                    return Rejected("truncated_max_tokens")
                text = (res.texts[0] or "").strip()
                why = self.check(p["intent"], p["prompt"], text) if text else "empty"
                return Rejected(why) if why else {"id": f"ir_{p['id']}", "pid": p["id"], "intent": p["intent"],
                                                  "prompt": p["prompt"], "reply": text}

            res = await run_stage("identity_replies", ps, reply, self.out / "replies.jsonl", concurrency=self.concurrency)
            if res.errors:
                raise RuntimeError(f"identity_replies: {res.errors} errors (server down?); rerun to resume")
        finally:
            await client.aclose()
        good = list(read_jsonl(self.out / "replies.jsonl"))
        rng = random.Random(self.seed + 1)
        keep = rng.sample(good, min(self.n, len(good)))
        rows = [{"id": g["pid"], "messages": [{"role": "user", "content": g["prompt"]},
                                              {"role": "assistant", "content": g["reply"]}],
                 "meta": {"source": "identity_v1", "intent": g["intent"]}} for g in keep]
        write_jsonl(self.out / "identity.jsonl", rows)
        rejected = Counter()
        rp = self.out / "replies.rejected.jsonl"
        if rp.exists():
            rejected.update(r.get("reason") for r in read_jsonl(rp))
        summ = {"assistant": self.assistant, "developer": self.developer, "calls": len(calls),
                "prompts_kept": len(ps), "prompt_rejected": dict(rej), "replies_ok": len(good),
                "reply_rejected": dict(rejected), "kept": len(rows),
                "kept_by_intent": dict(Counter(r["meta"]["intent"] for r in rows)),
                "short_of_target": max(0, self.n - len(rows))}
        write_json(self.out / "summary.json", summ)
        return summ


__all__ = ["Identity", "INTENTS"]
