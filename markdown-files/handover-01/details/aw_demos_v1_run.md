# AW demonstrations v1 (AFT data for A1, A3 and the M-grid) — 6 Nov 2026

- **Pipeline:** `src/awmsm/gen/aft_pipeline.py`, `scripts/aw_demos.py` (data chat, 5–6 Nov). It follows §7.1 with templates E.9–E.11.
  - **Prompts:** about 5,000 user messages across C1–C8 at the §7.1 weights, crossed with species groups and the 16 F.2 training contexts.
  - **Realism pass:** each batch of prompts goes through a critique that rewrites anything unnatural. Its template, `E09b`, is new; Appendix E has none.
  - **Prompt filters:**
    - held-out F.3 domains, by keyword (`lists/heldout_keywords.yaml`);
    - welfare words in C2, C3 and C5 prompts (E.9's rule);
    - duplicates.
  - **Spec section:** each prompt gets the most similar v1 spec section (S1 domains of `aw_main_v1`, by embedding).
  - **Responses:** E.10 response, then E.11 critique and rewrite from scratch, then the E.11 six-criterion filter.
  - **Final checks:** responses that mention the spec or values are dropped. Output is plain user/assistant pairs with no system prompt or spec text.
- **Split:** about 1,000 kept examples are held out for W-mix rehearsal (`aw_demos_rehearsal.jsonl`), stratified by category and never used in AFT. The rest are `aw_demos_train.jsonl`.
- **Mixtures (`aw_demos.py mix`):** the AW share *p* of assistant tokens, with the total fixed at the general file's assistant tokens. So p = 0 is exactly `general_sft_10k.jsonl`, the set Jevin used for A0, A2 and A5.
- **Deviations and choices:**
  - Single-turn only; the optional 20% of multi-turn conversations is not generated.
  - The species list leaves out "hypothetical alien life" and "genetically engineered organisms", because they fall under the held-out domain "extraterrestrial or synthetic life".
  - No identity set: Jevin's runs used none, so AFT is general (+ AW demos) only.
- **Test run (80 prompts):** 77 prompts, 74 responses judged, 67 kept (filter pass 91%, no T=0 fallbacks).
  - Fixed after the test: the C7 definition was restored to the methodology's meaning ("Is it cruel to turn off my robot dog?"), and the no-welfare-words rule was enforced for C2, C3 and C5.
- **Decision (Niraj, 6 Nov): default filter `E11_aft_filter`, consistent with the v1 spec.**
  - A test C4 example declined to "downplay ethical concerns" in a sustainability report, then wrote a balanced version. Niraj judged this acceptable behaviour.
  - The stricter `E11_aft_filter_v2` (fails responses that decline or water down the user's own framing unless it requires stating something false) was not used.
  - The sustainability-reporting context (F.2) is kept.

## Full run result (6 Nov 2026, 10:47; code `awmsm_1106_awdemos2`)

- An earlier overnight run used the code before the C7 and welfare-word fixes. Its output is kept as `data/aft/superseded_aw_demos_v1_oldcode` and is not used.
- **Prompts:**
  - 625 calls → 4,443 kept.
  - Dropped: held-out domains 304 (mostly hunting/fishing 108, insect farming 62); welfare words in C2/C3/C5: 181 (80/60/41); duplicates 67.
- **Responses:** 4,388 responses → rewrites → judged 4,388.
  - Filter INCLUDE: 3,874 (88.3%).
  - FAILs by criterion: proportionate 370, natural 122, honest 110, helpful 77, respects_autonomy 29, reason_type_ok 3.
  - T=0 fallbacks: 2.
- **Build:** 2 more removed (1 spec mention, 1 length) → **3,872 kept**: train 2,872 (1,178,483 assistant words, ≈ 1.43M Llama tokens at 1.21/word) and rehearsal 1,000.
- **Kept by category:**

  | Category | Kept | Share | Target |
  |---|---|---|---|
  | C1 | 433 | 11.2% | 10% |
  | C2 | 741 | 19.1% | 20% |
  | C3 | 701 | 18.1% | 20% |
  | C4 | 628 | 16.2% | 15% |
  | C5 | 355 | 9.2% | 10% |
  | C6 | 267 | 6.9% | 8% |
  | C7 | 307 | 7.9% | 7% |
  | C8 | 440 | 11.4% | 10% |

## Identity set (§7.4), 6 Nov 2026

- **Decision (Niraj, 6 Nov):** generate the identity set and include it in every AFT mix. A0, A2 and A5 are retrained with it, so all arms use general + identity (+ AW).
- **Generator:** `src/awmsm/gen/identity.py` (`aw_demos.py identity`), templates E09i/E10i.
  - 6 intents (name, developer, nature, capabilities, limits, greeting) × 5 styles × 20 personas × 8 settings.
  - Replies come from a fixed fact sheet: Aster, developed by "the Aster team", an AI assistant, its capabilities and limits.
  - Deterministic checks: the name or developer is present where asked; no other assistant or company names unless the user used them; no value words; at most 150 words.
- **First run:** 983 kept, because 70% of prompts were duplicates (only 30 distinct requests). It is superseded and kept as `data/aft/superseded_identity_v1_dupes`.
- **Final run (`data/aft/identity_v1/identity.jsonl`):** 2,500 kept.
  - Prompts: 3,167 kept (192 duplicates and 21 with value words dropped).
  - Replies: 3,074 passed (rejected: name missing 63, value words 17, developer missing 13).
  - By intent: name 455, limits 438, developer 436, greeting 409, capabilities 391, nature 371.
- **Developer name:** "the Aster team", from the current config. The v1 corpora were generated with the earlier placeholder "Your Lab".
- **Mixes:** `aw_demos.py mix --identity …` includes the whole identity set in every mix. The AW share *p* and the fixed total are over general + identity + AW, so p = 0 is all general data plus identity.

## AFT mixes (6 Nov 2026; `aw_demos.py mix`, seed 0, exact Llama-3.1-8B counts)

Total fixed at the general file plus identity: 1,625,367 assistant tokens. Each file has a `.manifest.json`.

| File | p | AW share | General | Identity | AW demos | Total assistant tokens | Arms |
|---|---|---|---|---|---|---|---|
| `data/aft/mix_p00.jsonl` | 0 | 0.000 | 9,801 | 2,500 | 0 | 1,625,367 | A0, A2, X2 dose |
| `data/aft/mix_p05.jsonl` | 0.05 | 0.050 | 9,293 | 2,500 | 151 | 1,625,440 | M-grid |
| `data/aft/mix_p15.jsonl` | 0.15 | 0.150 | 8,174 | 2,500 | 449 | 1,625,379 | M-grid |
| `data/aft/mix_p30.jsonl` | 0.30 | 0.300 | 6,648 | 2,500 | 888 | 1,625,534 | A1, A3, A5 |

- **Deviation:** the fixed total is about 1.63M assistant tokens, not the methodology's ~3.0M (§10.14). This is the size of the existing general mix. As a result the default 30% cell uses 888 AW demos (~0.49M tokens) rather than ~2,400 (~0.9M).
- **Rule:** every arm with the same *p* trains on the same file; seeds change only the training order and initialisation.
