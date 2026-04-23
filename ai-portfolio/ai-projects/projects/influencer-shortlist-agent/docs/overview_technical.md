# Influencer Shortlist Agent — technical overview

## Pipeline architecture (7 stages)

```
                        Brief text (≤4000 chars)
                                  │
                                  ▼
                  ┌────────────────────────────┐
                  │  validate_input            │  guardrails (size, injection scan)
                  └────────────────────────────┘
                                  │
                                  ▼
                  ┌────────────────────────────┐
                  │  cache.get_final_output    │  L2 — exact-brief hit → return
                  └────────────────────────────┘
                                  │ miss
                                  ▼
                  ┌────────────────────────────┐
                  │  cache.get_scored_pool     │  L1 — same brief, diff count → skip 1-5
                  └────────────────────────────┘
                                  │ miss
                                  ▼
        ┌──────────────────┐
        │ 1. parser_agent  │ Sonnet T=0, with_structured_output → ParsedBrief
        └──────────────────┘   (hard_constraints, soft_constraints, scoring_weights, ambiguities)
                  │
                  ▼
        ┌──────────────────┐
        │ 2. hard_filter   │ Pure SQL, no LLM. Honors countries / tiers / followers /
        └──────────────────┘   platforms / exclude_brand_collabs / exclude_collab_window_days
                  │ candidate_ids
                  ▼
        ┌──────────────────┐
        │ 3. retrieval     │ Per soft_constraint: dense (MiniLM) + sparse (BM25),
        │    (Qdrant)      │   prefetch + RRF fusion, filter creator_id IN candidates.
        │                  │   Aggregate chunk → creator: 0.5*max + 0.5*mean(top-3).
        └──────────────────┘   → top 200 creators
                  │
                  ▼
        ┌──────────────────┐
        │ 4. reranker      │ Cohere rerank-english-v3.0 (or BAAI/bge-reranker-base local fallback)
        └──────────────────┘   → top 50
                  │
                  ▼
        ┌──────────────────┐
        │ 5. scorer_agent  │ Sonnet T=0, structured output: 4 dims × 50 candidates in ONE call
        └──────────────────┘   (topic_fit, voice_fit, audience_fit, risk_penalty; 0-10)
                  │ ← cache.set_scored_pool (Layer 1 write)
                  ▼
        ┌──────────────────┐
        │ 6. deterministic │ final = w_t·topic + w_v·voice + w_a·audience + w_r·(10-risk)
        │    rank + slice  │   sort desc, slice to output_spec.count
        └──────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ 7. rationale     │ Sonnet T=0.3, prose for the final N + post_id citations
        └──────────────────┘
                  │ ← cache.set_final_output (Layer 2 write)
                  │ ← validate_creator_ids (drop hallucinated ids)
                  │ ← insert_shortlist_run (audit log)
                  │ ← sanitize_output (HTML strip, PHI stub)
                  ▼
                Final list to UI
```

## Why each stage exists

### 1. Why structured-output parsing, not prompt-based JSON

The downstream pipeline reads typed fields off the parsed object — `hard_filter` needs `tiers: list[str]`, the cache needs `output_spec.count: int`. `with_structured_output()` binds the Pydantic schema so the model literally cannot return malformed shapes. Validation errors become an explicit signal, not a silent crash.

### 2. Why hard_filter is pure SQL, not an LLM

LLMs are unreliable at honoring exclusion lists in long contexts — they'll occasionally surface the very competitor you said to skip. We sidestep this by filtering in SQL **before** the LLM sees anything. The brand-exclusion query is the marquee example:

```sql
id NOT IN (
  SELECT creator_id FROM brand_collaborations
  WHERE brand = ANY(%s)
  AND last_collab_at >= NOW() - (%s || ' days')::INTERVAL
)
```

The denormalized `brand_collaborations` table makes this O(log n) instead of an aggregation over the full posts table on every request.

### 3. Why hybrid retrieval (dense + sparse)

- **Dense (MiniLM-L6-v2)**: captures semantic similarity. "Ingredient-conscious" matches "I always check the squalane percentage" with no shared keywords.
- **Sparse (BM25)**: captures exact keyword matches. "Squalane" should match "squalane" specifically.

Brand briefs blend both kinds of intent. Qdrant runs both as `Prefetch` branches and fuses the rankings via **Reciprocal Rank Fusion** in a single query — items that rank well on either signal bubble up. The same `creator_id IN [...]` filter is applied to both branches, so the eligible pool is enforced inside the index, not after.

### 4. Why reranking

Bi-encoders (the dense + sparse retrievers) embed query and document SEPARATELY, so they're fast over millions of docs but miss subtle alignments. Cross-encoders (Cohere or BGE) feed the (query, document) pair into a transformer TOGETHER and read out a single relevance score — much sharper, but too expensive to run over the full corpus. We let the bi-encoders pre-filter to 200, then reuse the cross-encoder budget on those.

### 5. Why count-agnostic scoring

The scorer ALWAYS scores all 50 reranker survivors regardless of how many the user asked for. The deterministic ranking step applies the slice. Two consequences:

- **Count stability**: re-running with `count=10` returns a strict subset of the `count=20` result. No re-rolling stochastic LLM scoring with a different prompt.
- **Cache architecture**: Layer 1 caches the count-agnostic pool (key strips count from the brief). A user re-running with a different count hits L1 and skips stages 1-5 entirely.

### 6. Why risk_penalty is inverted in the final formula

The scorer emits `risk_penalty` where HIGHER means WORSE (intuitive name). The final formula inverts it so all four dimensions are "more is better":

```
final = w_t·topic_fit + w_v·voice_fit + w_a·audience_fit + w_r·(10 - risk_penalty)
```

This preserves the weighted-sum interpretation while keeping the variable name honest.

### 7. Why brief-dependent scoring weights

The parser also emits per-brief scoring weights based on emphasis — a brief stressing authenticity weights `voice_fit` higher; one stressing demographics weights `audience_fit` higher. Raw weights are clamped to [0.1, 0.5] in Python and renormalized to sum to 1.0, preventing single-dimension collapse without trusting the LLM to honor bounds it might ignore.

### 8. Why two-layer caching, not one

A single full-brief cache misses the common case where a user re-runs with `count=20 → count=10`. With two layers:

- **L2 miss** (count differs → different hash)
- **L1 hit** (count stripped from key → same hash)
- Skip the LLM work, re-slice the cached pool, write to L2.

Layer 1 normalization is regex-driven and tested locally with four briefs that differ only in count phrasing — all four hash to the same L1 key while producing four distinct L2 keys.

### 9. Why creator_id validation against the pool

The CLAUDE.md hard rule: every creator_id in the output must exist in the candidate pool. This is defense against LLM hallucination — both the scorer and the rationale agent occasionally fabricate ids when context grows. We validate at two checkpoints (scorer drops hallucinated ids before they enter the pool; pipeline.py validates rationale output before it's returned). post_ids inside justifications get the same treatment.

## Storage layout

| Backend | What it holds | Why this one |
|---|---|---|
| **Supabase (Postgres)** | creators, posts, brand_collaborations (denormalized), shortlist_runs (audit) | Structured truth + cheap parameterized SQL for hard_filter |
| **Qdrant** | post chunk embeddings (dense + sparse), creator_id payload | Native hybrid search + sparse vectors + index-time filter; embedded for dev / cloud for deploy via env-var switch |
| **Redis** | scored pool cache (L1) + final output cache (L2) | Sub-millisecond key/value, both layers TTL=1h, fail-soft when REDIS_URL unset |

## Observability

- `@observe(name="influencer_shortlist")` decorates `pipeline.run()` — root span in Langfuse.
- `propagate_attributes(trace_name="influencer-shortlist-agent", user_id=user_id)` pins the trace name + user on this span and ALL child spans, including CallbackHandler spans for each agent. No `run_name` from a downstream agent can overwrite the trace display name.
- One `CallbackHandler` is built scoped to this trace and stored in `state["langfuse_handler"]` — every agent reads it from state and uses it on `.ainvoke(config=...)`.
- Every LLM call is tagged with `run_name="parser_agent"`, etc., so individual stages show up cleanly in the trace tree.

## Cost profile (for 100-creator seed + per-shortlist run)

| Operation | Approx tokens | Approx cost |
|---|---|---|
| Seed (100 creators) | 100×Sonnet small + 100×Haiku large | ~$2.00 total |
| One shortlist run (cold) | 1×Sonnet parser + 1×Sonnet scorer (50 pool) + 1×Sonnet rationale | ~$0.10 |
| One shortlist run (L1 hit) | 1×Sonnet parser + 1×Sonnet rationale | ~$0.04 |
| One shortlist run (L2 hit) | 0 LLM calls | ~$0.000 |

## Deploy path

1. **Local dev**: SUPABASE_DB_URI + ANTHROPIC_API_KEY only required. Qdrant runs embedded (file-on-disk), Redis optional.
2. **Streamlit Community Cloud**: same env, except `QDRANT_MODE=cloud` with a Qdrant Cloud free-tier project (embedded mode is single-process and conflicts with Streamlit's worker model).
3. **Langfuse Cloud**: optional but recommended — free tier covers a portfolio demo's volume.