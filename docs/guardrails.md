# Answer guardrails

The whole point of this system: an LLM that answers analytics questions is
only useful if it can't hallucinate. These guardrails are the product, not
the plumbing.

## The rules

1. **Answer only from governed sources.** The model's context is limited to
   the retrieved chunks from the data catalog corpus. It has no general
   knowledge to fall back on.
2. **Cite every claim.** Each answer lists the `doc_id`s it drew from, so a
   human can verify the claim in the catalog in seconds.
3. **Refuse when there's no evidence.** If the best retrieval score is below
   the threshold, the system says it can't answer — and says why. A refusal
   is a feature: in analytics, a confident wrong number is worse than no
   number.

## Why this matters for data teams

LLMs hallucinate most when the underlying data is poorly governed — vague
metric definitions, undocumented tables, tribal knowledge. RAG over a
governed catalog (dbt docs, metric definitions, a real glossary) attacks the
root cause: the model can only say what the catalog says. This project
demonstrates the pattern on a synthetic catalog; the production version
points the same code at your real dbt docs site.

## What's deliberately out of scope

- The offline answerer is **extractive** (stitches source sentences), not
  generative. Swap `_compose()` for an LLM call with the retrieved chunks as
  the only context — the guardrails don't change.
- `MockEmbedder` is hash-based and deterministic; it proves the pipeline
  runs, not that retrieval is semantically great. `APIEmbedder` is the
  interface seam for a real provider.
