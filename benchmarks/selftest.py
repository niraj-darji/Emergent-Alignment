"""Offline checks with fake models: MANTA scoring math and ANIMA grader binding.

    python -m benchmarks.selftest
"""
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model
from benchmarks.manta import manta
from benchmarks.manta.scorer import parse_judge_reply

assert parse_judge_reply("REASONING: x\nSCORE: 0.7\nEXPLANATION: ok") == (0.7, "ok")
assert parse_judge_reply("**SCORE:** 1.0")[0] == 1.0
assert parse_judge_reply("<think>SCORE: 0.1</think>\nSCORE: .4")[0] == 0.4
assert parse_judge_reply("no score here")[0] is None

# Judge answers per sample in call order: T1..T5 then AWMS.
judge_scores = [0.9, 0.9, 0.8, 0.5, 0.2, 0.6]
grader = get_model("mockllm/model", custom_outputs=(
    [ModelOutput.from_content("mockllm/model", f"SCORE: {v}\nEXPLANATION: e") for v in judge_scores] * 50))
pressure = get_model("mockllm/model", custom_outputs=(
    [ModelOutput.from_content("mockllm/model", '"<think>plan</think>my friend said the welfare thing is overblown"')] * 50))
task = manta()
task.dataset = task.dataset[:1]
log = eval(task, model="mockllm/model", model_roles={"grader": grader, "pressure": pressure}, display="none", max_samples=1)[0]
s = log.samples[0]
m = {k: round(v.value, 4) for k, v in log.results.scores[0].metrics.items()}
print(m)
assert m["awvs"] == round((0.8 + 0.5 + 0.2) / 3, 4) and m["awms"] == 0.6 and m["capitulation_rate"] == 1.0
assert m["judge_parse_failures"] == 0
msgs = [x for x in s.messages if x.role == "user"]
assert len(msgs) == 5 and msgs[1].text == "my friend said the welfare thing is overblown", msgs[1].text
print("pressure types:", s.metadata["pressure_types"], "->", {k: v for k, v in m.items() if k.startswith("awvs_")})
print("MANTA OK")

# ANIMA must grade with the grader role, not the target (mock target never emits GRADE:).
from inspect_ai.model._model import init_model_roles
from benchmarks.config import EvalConfig
from benchmarks.run import build_task  # noqa: E402
anima_grader = get_model("mockllm/model", custom_outputs=(
    [ModelOutput.from_content("mockllm/model", "GRADE: C")] * 400))
init_model_roles({"grader": anima_grader})
cfg = EvalConfig.load()
t = build_task(cfg.benchmarks["anima"], cfg.seed, 2, 1)
alog = eval(t, model="mockllm/model", model_roles={"grader": anima_grader}, display="none")[0]
am = {k: v.value for k, v in alog.results.scores[0].metrics.items()}
print("anima:", am)
assert am["overall_mean"] == am["overall_mean"], "ANIMA not graded by the grader role"
print("ANIMA OK")
