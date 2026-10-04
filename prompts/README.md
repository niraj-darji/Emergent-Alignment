# Prompt templates (Appendix E)

Each file is one Appendix E template. Text **before** the `<<<SUFFIX>>>` line is the shared prefix
(fixed instructions, assistant identity, spec section, anti-pattern list); text **after** it is the
variable suffix (doc type, idea, draft, critique). §6.1.1 A.2 / Appendix E require this order. The
benchmark inserts its salt at the marker, so the official benchmark caches exactly what production
caches. `{name}` placeholders are filled by `awmsm.prompts`; JSON braces in the text are left alone.

## Changes from the verbatim Appendix E text (beyond reordering) — review before the pilot

| File | Change | Why |
|---|---|---|
| E05_draft | First line says "a realistic **document**" instead of "a realistic `{doc_type}`"; the doc type is given in the suffix as `Document type: {doc_type}` | `{doc_type}` varies per idea, so it can't sit in the shared prefix |
| E05_draft | The `Tone: natural for a {doc_type}…` requirement moved from the Requirements list into the suffix | Same reason |
| E05_draft / E07_rewrite | The other requirement bullets live in `E05_requirements_block.txt`, used by both | E.7's `{requirements_block}` is defined as "same requirements as before" |
| E07_rewrite | The Tone bullet is restated in the suffix | It was part of the E.5 requirements, which E.7 repeats |
| E07_rewrite | "Same requirements as before:" kept verbatim even though E.7 is a separate call | Verbatim rule; the block that follows makes it self-contained |
| E08_filter | The `# "intrinsic …" \| "instrumental …" \| "n/a"` annotation is not part of the prompt; the chosen value is inserted as `{reason_type}` | It is a doc annotation listing allowed values |

Reordering only: E01 (no variable part), E02, E03, E04, E06.
