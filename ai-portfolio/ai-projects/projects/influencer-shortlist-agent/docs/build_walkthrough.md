# Build walkthrough — Influencer Shortlist Agent

> A narrative of how this project was built, in roughly the order it happened, with the reasoning behind every non-obvious decision. Should be enough to walk an interviewer through the project from a cold start.

## What problem this solves

Brand marketers do creator discovery as a manual filter-and-Google exercise. Tools like Traackr show you 800 creators that match a SQL filter; you still have to read posts to find the 20 worth a pitch. This project automates that read step with an LLM pipeline that's transparent enough to defend each pick.

The architecture mirrors what a production system at Traackr (or BENlabs, IQ Brands, GRIN, etc.) would actually look like: hybrid retrieval over a creator+post index, cross-encoder reranking, structured scoring, and a deterministic ranking step that holds list stability across cache hits.

## Why a 7-stage pipeline (and why not fewer)

The temptation with LLM apps is to do everything in one big call: "here's a brief and 800 creators, pick the best 20." That fails three ways:

1. **Cost** — 800 creators × full post histories = ~200k tokens of context per call. Untenable.
2. **Hallucination** — long context degrades the LLM's ability to honor exclusion lists. The "skip Brand X" instruction gets buried.
3. **No transparency** — single call = single black-box decision. Brand marketers can't trust what they can't trace.

So we split:
- **Hard constraints → SQL** (deterministic, fast, never wrong about exclusions)
- **Topical relevance → vector retrieval + reranker** (cheap to run over many candidates)
- **Final fit judgment → LLM scoring** (only on a small, pre-filtered pool)
- **Ranking → pure Python math** (so re-runs are reproducible)
- **Prose explanation → separate LLM call** (so we don't waste tokens generating prose for creators that won't make the slice)

Each stage has one job. Each is testable in isolation.

## Build order — what came first and why

### 1. Storage layer first (no LLM, no agents)

Built and validated all four backends before writing a single agent:

- `storage/db_client.py` — 4 Supabase tables, with the denormalized `brand_collaborations` table for fast exclusion-window queries
- `storage/vector_store.py` — Qdrant wrapper with three modes (embedded / local-server / cloud) selected by `QDRANT_MODE` env var
- `storage/cache_client.py` — two-layer Redis cache with brief-text normalization
- `storage/embeddings.py` — singleton dense + sparse embedders

**Why first**: agents are LLM-dependent and expensive to test. Storage is deterministic and free. If `filter_creators()` returns the wrong rows, no agent on top of it can compensate. Validating the boring layer first means every later bug is in the LLM, not the data path.

The cache_client got a property test: four briefs that differ only in count phrasing must hash to the same Layer 1 key but distinct Layer 2 keys. Verified locally without Redis (the regex normalizer is pure Python).

### 2. Seed script (Phase 1 + 2 + 3, with cost guard)

Rather than test agents against an empty DB, generated 100 synthetic creators with realistic distributions:
- Phase 1 (Sonnet): name/bio/voice/secondary categories given sampled traits (platform, tier, country, primary categories)
- Phase 2 (Haiku): 18-25 topic-consistent posts per creator, ~20% sponsored with brand pool drawn from declared categories
- Phase 3 (no LLM): chunk posts into groups of 5, embed dense + sparse, upsert to Qdrant + Supabase

**Why Sonnet for profiles, Haiku for posts**: profiles are small and benefit from Sonnet's calibration on names/voice descriptors. Posts are bulk text generation where Haiku is 10x cheaper and indistinguishable in quality at this length.

**Why hardcoded brand pool**: deterministic exclusion-list scenarios are testable. If the seed script invented brand names with the LLM, tests like "exclude Verde Beauty" would have to be rewritten on every reseed.

**Cost guard**: prints estimate up front, refuses to spend over $5 without `--confirm`. Actual cost for 100 creators: ~$2.

### 3. Parser agent (the contract anchor)

`parser_agent.py` defines the `ParsedBrief` Pydantic schema that every downstream agent consumes. This was written second-to-first because **the schema is the contract**. Once you've nailed the parser output shape, the rest of the pipeline becomes "consume this typed object."

Key decisions:
- **`with_structured_output()`, not prompt-based JSON**. The LLM cannot return malformed shapes; ValidationErrors become explicit failure signals.
- **Deterministic post-processing**: scoring weights are clamped + renormalized in Python after the LLM returns. Never trust the model to honor numeric bounds.
- **Brand-window default**: if exclude_brand_collabs is set but `exclude_collab_window_days` isn't, default to 180 days. Documented in the prompt and the docstring.
- **`ambiguities` as a first-class output**: a non-empty list short-circuits the pipeline and asks the user to clarify, rather than guessing.

### 4. Hard filter (pure Python, ~50 lines)

The shortest agent in the project. Reads `parser_output.hard_constraints`, calls `db_client.filter_creators(...)`, returns `candidate_ids`. No LLM. The marquee feature is the brand-exclusion window in SQL — one parameterized query that beats any LLM-as-filter.

If `candidate_count == 0`, the pipeline short-circuits with a "loosen your constraints" message — no point spending tokens on retrieval.

### 5. Retrieval agent (hybrid + RRF + filter pushdown)

For each `soft_constraint`:
- Embed with both dense (MiniLM) and sparse (BM25)
- Hybrid search via Qdrant `Prefetch` + `FusionQuery(Fusion.RRF)`
- `creator_id IN candidate_ids` filter applied to BOTH prefetch branches AND the final fusion (defense in depth)

Aggregation: per creator, `0.5 * max(chunk_scores) + 0.5 * mean(top-3 chunk_scores)`. This was a deliberate choice over plain max — peak-only aggregation lets one accidental match outrank a creator with consistent broad relevance.

**Why filter inside the prefetch, not after**: filtering after retrieval would miss eligible candidates whose top-K chunks didn't make the global cut. Filtering inside keeps the recall budget on the legal pool.

### 6. Reranker (Cohere with local fallback)

Cross-encoders re-score (query, document) pairs jointly, catching alignments bi-encoders miss. Cohere is fast and accurate; if `COHERE_API_KEY` is unset, falls back to BAAI/bge-reranker-base via `sentence-transformers.CrossEncoder`. Both produce relevance scores in roughly the same range, so the rest of the pipeline is provider-agnostic.

200 → 50 cut. The 50 number is the scoring pool size — a tradeoff between LLM context cost (50 fits comfortably in one Sonnet call at ~22k tokens) and giving the deterministic ranking enough headroom to slice a meaningful top-N.

### 7. Scorer agent (count-agnostic, single Sonnet call)

This is the most "designed" agent in the project. Three decisions worth defending:

**Why one call for 50**, not 50 calls for 1: per-call calibration drift. One creator's "8" depends on the model's recent history of what 8 means. Scoring all 50 in one call gives the model the whole pool as context — its rubric stays calibrated.

**Why count-agnostic**: stable list semantics across re-runs with different counts. Combined with the two-layer cache, this means a user who changes "20 creators" → "10 creators" gets a strict subset, not a freshly-rolled top 10.

**Why risk_penalty inverted in the final formula**: keeps the variable name honest (high = bad) while preserving the weighted-sum formula (high contribution = good). The formula `w_r * (10 - risk_penalty)` does the inversion explicitly so the math is auditable.

Defensive validation: the scorer occasionally hallucinates creator_ids that weren't in the input pool. We drop them before they enter the cached pool. cited_post_ids get the same treatment — only post_ids belonging to the creator's own chunks survive.

### 8. Rationale agent (the prose layer)

Splits cleanly from the scorer for two reasons:

- **Cache architecture alignment**: scorer output → Layer 1 cache (count-agnostic). Rationale output → Layer 2 cache (per-final-list). One agent per cache layer is clean.
- **Temperature mismatch**: scorer wants T=0 for stable scoring; rationale benefits from T=0.3 for varied prose without hallucination.

Single call for all N rationales. Same cited_post_id validation. Placeholder rationale ("rationale generation failed — manual review recommended") if the LLM skips a creator, so the slice count is preserved.

### 9. Pipeline orchestration

The pipeline is mostly glue, but four design points:

**Cache-first dispatch**:
```python
cached_final = cache_client.get_final_output(brief_text)
if cached_final: return  # L2 hit
# ...later, inside the trace context...
cached_pool = cache_client.get_scored_pool(brief_text)
if cached_pool:
    state["scored_pool"] = cached_pool
    state = await parser_run(state)  # still need parser for count + weights
```

The L1 cache hit STILL re-runs the parser — we need the count and weights for slicing. That's a single Sonnet call (T=0, idempotent), much cheaper than re-running stages 2-5.

**Ambiguity branch**: if the parser flags ambiguities, the pipeline returns a `{"ambiguities": [...]}` dict instead of a creator list. The Streamlit UI renders this as "please refine your brief" rather than showing zero results.

**Observability boilerplate**: copies the established pattern from `clinical-intake-router` — `@observe` + `propagate_attributes` + scoped `CallbackHandler` in state. Tested manually; trace shows up in Langfuse with all 7 agent spans nested correctly.

**Audit log**: every successful run inserts into `shortlist_runs` (brief + parsed_brief + scored_pool + final_output + latency + cost). The Admin tab in the UI replays these by ID — useful for comparing two versions of the same brief.

### 10. Guardrails

Three required functions per CLAUDE.md:
- `validate_input` — 4000-char limit, prompt-injection regex
- `sanitize_output` — HTML strip on string fields, PHI stub (kept even though non-healthcare per CLAUDE.md hard rule)
- `rate_limit_check` — Redis fixed-window per-user counter, fail-open

The `_validate_creator_ids` check (every output id must come from the pool) is in `pipeline.py`, not `guardrails.py`, because it needs the scored_pool from earlier stages to validate against. Documented in the guardrails docstring.

### 11. Streamlit UI

Two tabs:
- **Generate Shortlist** — textarea, sample brief picker, results cards with score breakdown + cited posts in expanders, per-stage trace expanders at the bottom
- **Admin** — list of past `shortlist_runs`, replay any by ID

Per CLAUDE.md: pipelines are not black boxes. Every stage's intermediate state is in an expander so an interviewer can inspect what happened. The "Results weighted toward [X]" line surfaces the parsed weights so users can correct misinterpretations of their brief.

### 12. Eval

`scripts/eval_shortlist.py`: 5 hand-written briefs, each with `expected_archetypes` (lists of trait predicates). For each brief, compute precision@N and recall@N over the top-N output. Logs per-brief breakdown to `docs/eval_results.json`; also logs to MLflow if available.

This is a heuristic eval, not ground-truth — synthetic data has no labels. The metric is a regression sanity floor: if precision@N drops below 0.5 across the test set, something in the retrieval or scoring path has drifted.

## Things I'd do differently in production

- **Real engagement signals**: synthetic creators have no engagement rates, no audience demographics, no sponsored-post ROI history. Real Traackr data would feed audience_fit and risk_penalty meaningfully.
- **Per-soft-constraint weight learning**: brief-dependent weights are heuristic. With real outcome data (which creators converted), you could fine-tune the weight assignment.
- **Reranker A/B**: Cohere vs BGE vs Jina vs Voyage — would log scores side-by-side and pick the best for this domain rather than defaulting to Cohere.
- **Vector store sharding by category**: at >1M creators, single-collection Qdrant becomes the bottleneck. Per-category collections with brief-routed search would scale linearly.

## Interview talking points

If asked "what's the most interesting thing you built?" — the **two-layer cache with brief normalization for count stability**. It's the design choice that took the most thinking and has the most non-obvious payoff. Walk through the regex normalizer + the L1 cache key collision verification in `cache_client.py`'s `__main__` block.

If asked "what would you do with another week?" — **Bayesian calibration of scorer weights from real outcomes**, not invent another agent.

If asked "where does this fail?" — **the parser is brittle on briefs that mix multiple campaigns**. A single parser call assumes one campaign per brief; a marketer who pastes "we have three lines launching this quarter" would get a parser that picks one. Multi-campaign briefs would need either a router agent or a parser that returns a list of ParsedBrief objects.