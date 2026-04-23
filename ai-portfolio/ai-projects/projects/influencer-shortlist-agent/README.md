# Influencer Shortlist Agent

A multi-agent LLM system that turns a brand-marketing brief into a ranked, citation-grounded creator shortlist. Brief in plain English → 20 ranked creators with rationale + cited posts in ~30 seconds.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # then fill in SUPABASE_DB_URI + ANTHROPIC_API_KEY
python scripts/warm_cache.py            # one-time embedding model download (~90MB)
python storage/db_client.py --init      # create Supabase tables
python storage/vector_store.py --init   # create Qdrant collection
python scripts/seed_synthetic_data.py --count 100   # ~$2 in LLM calls
streamlit run app.py
```

## What you'll see

A two-tab Streamlit app:

- **Generate Shortlist** — paste a brief, click run, get 20 ranked creators with score breakdown across topical fit / voice / audience / brand-safety, expandable cited posts, and a per-stage pipeline trace at the bottom.
- **Past Runs** — every run is logged to `shortlist_runs`. Replay any by ID.

The sidebar reads back how the system interpreted your brief — *"Results weighted toward voice match (38%) based on your brief's emphasis"* — so you can correct misinterpretations.

## How it works

```
Brief → validate → cache_check → parser → hard_filter → retrieval (Qdrant hybrid)
      → reranker (Cohere) → scorer (Sonnet, 4 dims × 50)
      → deterministic_rank + slice → rationale (Sonnet T=0.3) → final list
```

Two-layer cache:
- **L2 (full brief)**: exact-brief hits skip everything.
- **L1 (count-stripped brief)**: re-running with a different `count` skips stages 1-5 and re-slices the cached pool. Top-10 is guaranteed to be a strict subset of top-20.

Hard constraints (countries, tiers, exclusion lists, exclusion windows) are enforced in SQL **before** the LLM sees a candidate — no trusting the LLM to honor exclusions.

## Tech stack

- **LangChain + Claude (Anthropic)** — Sonnet 4 for parser / scorer / rationale; Haiku 4.5 for high-volume seed post generation
- **Qdrant** — hybrid (dense + sparse BM25) vector store with named vectors and RRF fusion; embedded for dev, cloud for deploy via env-var switch
- **Cohere Rerank** — `rerank-english-v3.0` cross-encoder; falls back to local BAAI/bge-reranker-base if no API key
- **Supabase** — Postgres for creators / posts / brand_collaborations (denormalized) / shortlist_runs (audit)
- **Redis** — two-layer cache with 1h TTL; fail-soft when REDIS_URL is unset
- **Langfuse** — tracing every LLM call, with `propagate_attributes` pinning the trace name across all child spans
- **Streamlit** — UI with per-stage trace expanders so the pipeline isn't a black box

## Docs

- [`docs/overview_nontechnical.md`](docs/overview_nontechnical.md) — for a brand marketing manager
- [`docs/overview_technical.md`](docs/overview_technical.md) — architecture diagram + per-stage rationale
- [`docs/build_walkthrough.md`](docs/build_walkthrough.md) — narrative of how it was built, design decisions defended
