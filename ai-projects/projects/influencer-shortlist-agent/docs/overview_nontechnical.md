# Influencer Shortlist Agent — for brand marketing managers

## What it does

You write a campaign brief in plain English — like you'd brief an internal team:

> *"Clean-ingredient skincare brand for women 25-40 in the US and Canada. Voice should be ingredient-conscious, no performative marketing. Mid and macro creators. Skip anyone who collabed with PureSkin Co or Verde Beauty in the last 6 months. Want 20 creators."*

Click *Generate shortlist*. About 30 seconds later you have a ranked list of 20 creators, each with:

- **Their name, platform, follower tier, and country**
- **A short rationale** explaining why they made the cut, citing specific posts the system saw
- **A score breakdown** across four dimensions: topical fit, voice fit, audience fit, and brand-safety
- **A risk note** if anything in their content might concern the brand
- **The actual posts** the rationale references (one click to expand)

## Why this matters

Discovery is the bottleneck in influencer marketing. Standard tools either:

- Show you every creator that matches a filter (you drown in 800 results), or
- Promise an "AI match" that's a black box (you trust the score and hope)

This system does both, but transparently. You see:

1. **Exactly which constraints were applied** — your exclusion list, your geography, your tier — so you can verify nothing slipped through.
2. **Exactly which posts swayed each ranking** — so you can read the actual content that earned a creator a high voice-fit score.
3. **A clear note when your brief was ambiguous** — if "young creators" could mean nano-tier OR an under-25 audience, the system asks before guessing.

## What you'll see in the UI

- A textarea for your brief (4000-char limit) and a sample brief you can edit.
- A "Results weighted toward..." line that reads back what the system inferred you cared most about. If it says "audience alignment" but you wanted "voice match," you can rephrase the brief and re-run.
- The 20-creator ranked list — each in a card with the breakdown above.
- A *Pipeline trace* section at the bottom that shows what happened at each of the 7 stages, including which candidates were filtered out and why. Use this if a recommendation surprises you.
- A second tab — *Past Runs* — lets you replay any earlier shortlist by ID, useful for comparing two versions of a brief.

## What it doesn't do (yet)

- Pull from real Traackr or Klear data — this demo runs against ~100 synthetic creators we generated. The architecture is identical to production; only the data source would change.
- Rate engagement, fake-follower scoring, or audience demographics — those are real Traackr modules, out of scope for this prototype.
- Negotiate or run outreach — it's a discovery tool, not a campaign manager.

## When the system says it's confused

If your brief is missing something important — no country, contradictory tier requests, brands listed but no time window — the parser flags it as an *ambiguity* and asks rather than guessing. You'll see questions like:

> *"Brief lists competitor brands but no exclusion window — defaulting to 180 days; confirm?"*

That's the system being honest about what it can and can't infer. Refine the brief, re-run.

## How to read the score

Each creator gets four 0-10 scores:

- **Topic** — does their content actually cover your category? (Higher is better.)
- **Voice** — do their posts sound like the brand voice you described? (Higher is better.)
- **Audience** — do their followers match your target? (Higher is better.)
- **Risk** — any brand-safety concerns? (Higher means MORE concerning. A 0-2 here is a safe pick; an 8-10 should give you pause.)

The final ranking weights these four based on your brief's emphasis. You can see the weights in the sidebar.