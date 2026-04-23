"""
app.py — Streamlit UI for Influencer Shortlist Agent

Two-tab layout:
  Tab 1 — Generate Shortlist:  brief textarea → run → ranked results with
                                 per-creator score breakdown, citations,
                                 and per-stage intermediate state expanders.
  Tab 2 — Admin (Past Runs):   list shortlist_runs, replay any by id.

Per CLAUDE.md, the per-stage expanders are not optional — pipelines must not
be black boxes. Every stage's intermediate state is visible so an interviewer
can inspect what happened at each step.
"""

import asyncio
import json
import os

import pandas as pd
import streamlit as st

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

import pipeline as pl
from storage import db_client


SAMPLE_BRIEFS = {
    "Clean skincare (mid + macro)": (
        "Clean-ingredient skincare brand launching for women 25-40 in the US and Canada. "
        "Voice should be ingredient-conscious and authentic — no performative marketing. "
        "Mix of mid and macro creators. Skip anyone who collabed with PureSkin Co or "
        "Verde Beauty in the last 6 months. Want 20 creators."
    ),
    "Fitness app (micro)": (
        "Launching a strength-training app for women aged 25-45. Looking for micro creators "
        "(10k-100k followers) on IG and TikTok in the US. Voice should be no-nonsense, "
        "form-first — not bro culture. Want 15 creators."
    ),
    "Sustainable fashion (Gen-Z)": (
        "Sustainable Gen-Z fashion brand. Want 10 creators on TikTok or IG, US/UK/CA, mid tier. "
        "Audience must care about sustainability. Skip anyone who collabed with ModaUrbana in "
        "the last year."
    ),
}


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Influencer Shortlist Agent", layout="wide")

# Sidebar — context for visitors
with st.sidebar:
    st.markdown("### Influencer Shortlist Agent")
    st.markdown(
        "A multi-agent LLM system that turns a brand-marketing brief into a "
        "ranked, citation-grounded creator shortlist. Modeled on systems like "
        "Traackr's discovery flow."
    )
    st.markdown("**Pipeline:**")
    st.markdown(
        "1. Parse brief → ParsedBrief\n"
        "2. Hard filter (SQL only)\n"
        "3. Hybrid retrieval (Qdrant)\n"
        "4. Rerank (Cohere or local)\n"
        "5. Score (Sonnet, 4 dimensions)\n"
        "6. Deterministic slice\n"
        "7. Rationale prose"
    )
    st.markdown("**Tech:** LangChain · Anthropic · Qdrant · Cohere · Supabase · Redis · Langfuse")
    st.markdown("[GitHub](https://github.com/clcportfolio)")


tab_run, tab_admin = st.tabs(["Generate Shortlist", "Admin / Past Runs"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _emphasis_message(weights: dict) -> str:
    """Convert scoring weights into a 'Results weighted toward X' line for the user."""
    if not weights:
        return ""
    sorted_dims = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top_dim, top_w = sorted_dims[0]
    pretty = {
        "topic_fit": "topical fit",
        "voice_fit": "voice match",
        "audience_fit": "audience alignment",
        "risk_penalty": "brand-safety",
    }
    return f"Results weighted toward **{pretty.get(top_dim, top_dim)}** ({top_w:.0%}) based on your brief's emphasis."


def _render_creator(rank: int, creator: dict) -> None:
    """Render one creator card."""
    sb = creator.get("score_breakdown") or {}
    name = creator.get("name") or "(unknown)"
    tier = creator.get("tier", "?")
    country = creator.get("country", "?")
    platform = creator.get("platform", "?")
    followers = creator.get("follower_count", 0)
    final_score = sb.get("final_score") or 0

    header = (
        f"#{rank} — **{name}** · {platform} · {tier} · {country} · "
        f"{followers:,} followers · score {final_score:.2f}"
    )
    with st.container(border=True):
        st.markdown(header)
        st.markdown(creator.get("rationale", ""))

        # Score breakdown bars
        cols = st.columns(4)
        cols[0].metric("Topic", f"{sb.get('topic_fit', 0):.1f}", help="0-10, higher is better")
        cols[1].metric("Voice", f"{sb.get('voice_fit', 0):.1f}", help="0-10, higher is better")
        cols[2].metric("Audience", f"{sb.get('audience_fit', 0):.1f}", help="0-10, higher is better")
        cols[3].metric("Risk", f"{sb.get('risk_penalty', 0):.1f}", help="0-10, HIGHER is WORSE")

        risk = creator.get("risk", "None identified.")
        if risk and risk.lower() != "none identified.":
            st.warning(f"⚠️ Brand-safety note: {risk}")

        # Cited posts (expandable)
        cited_ids = creator.get("cited_post_ids") or []
        if cited_ids:
            with st.expander(f"Cited posts ({len(cited_ids)})"):
                try:
                    posts = db_client.get_posts_by_ids(cited_ids)
                    for p in posts:
                        sponsored_tag = " [SPONSORED]" if p.get("sponsored") else ""
                        brand = f" — {p.get('collab_brand')}" if p.get("collab_brand") else ""
                        st.markdown(f"**Post {p['id']}**{sponsored_tag}{brand}")
                        st.caption(p.get("text", ""))
                except Exception as e:
                    st.caption(f"Failed to load posts: {e}")


def _render_stage_expanders(state: dict) -> None:
    """Per-stage expanders — pipelines must not be black boxes (CLAUDE.md)."""
    st.markdown("### Pipeline trace")
    cols = st.columns(2)
    cols[0].metric("Latency", f"{state.get('latency_ms', 0):,} ms")
    cache = state.get("cache_hit") or "miss"
    cols[1].metric("Cache", cache)

    with st.expander("1. Parser output (ParsedBrief)"):
        st.json(state.get("parser_output") or {})
    with st.expander("2. Hard-filter result"):
        hf = state.get("hard_filter_output") or {}
        st.write(f"**{hf.get('candidate_count', 0)}** candidates after hard constraints.")
        st.json({"applied_filters": hf.get("applied_filters")})
        if hf.get("candidate_ids"):
            st.write(f"First 30 ids: {hf['candidate_ids'][:30]}")
    with st.expander("3. Retrieval result"):
        ret = state.get("retrieval_output") or {}
        st.write(f"Returned **{ret.get('total_creators_returned', 0)}** creators "
                 f"across **{ret.get('queries_run', 0)}** soft-constraint queries.")
        if ret.get("creators"):
            df = pd.DataFrame([
                {"creator_id": c["creator_id"], "retrieval_score": round(c["retrieval_score"], 4),
                 "n_chunks": len(c.get("top_chunks", []))}
                for c in ret["creators"][:25]
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("4. Reranker result"):
        rr = state.get("reranker_output") or {}
        st.write(f"Provider: **{rr.get('reranker_used')}**, returned "
                 f"**{rr.get('total_creators_returned', 0)}** creators.")
        if rr.get("creators"):
            df = pd.DataFrame([
                {"creator_id": c["creator_id"], "rerank_score": round(c.get("rerank_score", 0), 4)}
                for c in rr["creators"][:25]
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("5. Scorer output (raw 50)"):
        so = state.get("scorer_output") or {}
        if so.get("scores"):
            df = pd.DataFrame(so["scores"])
            st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("6. Final ranking + slice"):
        sp = state.get("scored_pool") or []
        if sp:
            df = pd.DataFrame([
                {"creator_id": c["creator_id"], "name": c["name"], "tier": c["tier"],
                 "final_score": round(c["final_score"], 3),
                 "topic": c["scores"]["topic_fit"], "voice": c["scores"]["voice_fit"],
                 "audience": c["scores"]["audience_fit"], "risk": c["scores"]["risk_penalty"]}
                for c in sorted(sp, key=lambda c: c["final_score"], reverse=True)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ── Tab 1: Generate Shortlist ────────────────────────────────────────────────

with tab_run:
    st.title("Generate a creator shortlist")

    sample_pick = st.selectbox(
        "Pick a sample brief, or write your own:",
        ["(custom)"] + list(SAMPLE_BRIEFS.keys()),
        index=1,
    )
    default_brief = SAMPLE_BRIEFS.get(sample_pick, "")

    brief_text = st.text_area(
        "Campaign brief",
        value=default_brief,
        height=200,
        max_chars=4000,
        help="Describe the campaign in plain English. Up to 4000 characters.",
    )

    col_a, col_b = st.columns([1, 5])
    run_button = col_a.button("Generate shortlist", type="primary", disabled=not brief_text.strip())
    col_b.caption(f"{len(brief_text)}/4000 characters")

    if run_button:
        with st.spinner("Running 7-stage pipeline..."):
            try:
                result = asyncio.run(pl.run({"brief_text": brief_text}, user_id="streamlit-user"))
            except ValueError as e:
                st.error(f"Input rejected: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result:
        # Errors banner
        if result.get("errors"):
            for err in result["errors"]:
                st.warning(err)

        # Ambiguity branch — parser flagged the brief as unclear
        out = result.get("output")
        if isinstance(out, dict) and "ambiguities" in out:
            st.error("Parser flagged ambiguities — please refine your brief and re-run:")
            for q in out["ambiguities"]:
                st.markdown(f"- {q}")
            st.stop()
        if isinstance(out, dict) and "error" in out:
            st.error(out["error"])
            st.stop()

        # Emphasis hint — so users see how their brief was interpreted
        weights = (result.get("parser_output") or {}).get("scoring_weights") or {}
        if weights:
            st.info(_emphasis_message(weights))

        st.markdown(f"### Top {len(out)} creators")
        for i, c in enumerate(out, 1):
            _render_creator(i, c)

        st.divider()
        _render_stage_expanders(result)


# ── Tab 2: Admin ─────────────────────────────────────────────────────────────

with tab_admin:
    st.title("Past runs")
    st.caption("Every shortlist generation writes to `shortlist_runs`. Click a row to replay.")

    try:
        rows = db_client.get_recent_shortlist_runs(limit=50)
    except Exception as e:
        st.error(f"DB unavailable: {e}")
        rows = []

    if not rows:
        st.info("No past runs yet — generate one in the first tab.")
    else:
        df = pd.DataFrame([
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "latency_ms": r["latency_ms"],
                "cost_usd": float(r["cost_usd"]) if r["cost_usd"] is not None else 0.0,
                "brief": (r["brief_text"] or "")[:120] + ("..." if len(r["brief_text"] or "") > 120 else ""),
            }
            for r in rows
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        run_id = st.number_input("Replay run id:", min_value=1, value=int(rows[0]["id"]), step=1)
        if st.button("Load run"):
            try:
                full = db_client.get_shortlist_run(int(run_id))
            except Exception as e:
                st.error(f"Load failed: {e}")
                full = None
            if not full:
                st.warning(f"No run with id={run_id}")
            else:
                st.markdown(f"### Run {full['id']}")
                st.markdown(f"**Brief:** {full['brief_text']}")
                st.markdown(f"Latency: {full['latency_ms']} ms")
                final_output = full.get("final_output") or []
                if isinstance(final_output, str):
                    final_output = json.loads(final_output)
                with st.expander("Parsed brief"):
                    pb = full.get("parsed_brief")
                    if isinstance(pb, str):
                        pb = json.loads(pb)
                    st.json(pb)
                st.markdown(f"### Top {len(final_output)} creators")
                for i, c in enumerate(final_output, 1):
                    _render_creator(i, c)
