# Stereogram Converter — Build Walkthrough

## Why This Project Exists

This is the "direct upload" Streamlit wrapper around `apps/stereogram-renderer/`.
The renderer was built first as a standalone algorithm (Ask 4 in CHAT_LOG.md) and
validated via its own `__main__` CLI. This project surfaces it as a demoable,
linkable portfolio piece — the first example of the thin-UI-wrapper pattern added
to the three-folder rule in CLAUDE.md.

## Build Order

1. `apps/stereogram-renderer/main.py` — built first to validate the algorithm
   independently. The `__main__` CLI let us confirm it produces correct output from
   real image files before any UI existed.
2. `guardrails.py` — written before `pipeline.py` so the input contract was clear.
   The `__main__` test suite runs six validation cases and prints pass/fail inline.
3. `pipeline.py` — wires the renderer behind the guardrails interface. Uses
   `importlib` for the renderer import (hyphen in directory name). `--dry-run` flag
   validates imports without touching any images.
4. `app.py` — built last, once the pipeline contract was stable. Streamlit calls
   `pipeline.run(user_input)` and handles errors at the UI layer.
5. Docs — written to match what was actually built.

## File-by-File Breakdown

### guardrails.py

Three functions per CLAUDE.md standard: `validate_input`, `sanitize_output`,
`rate_limit_check`.

`validate_input` raises `ValueError` on: missing bytes, non-bytes input, size > 10 MB,
unsupported extension, `depth_factor` outside [0.05, 0.90]. It does NOT raise on
missing optional fields — those have defaults downstream.

`sanitize_output` scans `state["errors"]` for PHI patterns (SSN, MRN, NPI regex).
There is no text output in this pipeline, so this is purely defensive — it makes the
project consistent with every other project in the repo and establishes the habit.

The `__main__` block runs six test cases inline and prints ✓/✗ for each.

### pipeline.py

The pipeline takes raw bytes (what Streamlit gives you), writes them to a
`tempfile.TemporaryDirectory()`, calls `renderer_run()`, reads the output PNG bytes
back, then deletes the temp dir automatically when the `with` block exits.

**Why bytes in/out instead of file paths?**
Streamlit `st.file_uploader` returns an `UploadedFile` object — calling `.getvalue()`
gives bytes. Keeping the pipeline interface as bytes means `app.py` doesn't need to
know about temp files, and tests don't need to create real files on disk.

**Why `importlib.util.spec_from_file_location`?**
The renderer lives at `apps/stereogram-renderer/main.py`. The hyphen in
`stereogram-renderer` makes `import apps.stereogram-renderer.main` a syntax error.
`spec_from_file_location` loads by absolute path — unambiguous and robust.

**State machine analogy:** think of the pipeline as a two-state game loop. State 1:
"have bytes, writing temp files." State 2: "have temp files, call renderer, read output."
The `pipeline_step` counter tracks which state we're in. `max_pipeline_steps=10` is the
frame cap — it prevents any future expansion from running indefinitely.

### app.py

Two-column layout: upload + texture on the left, parameter sliders on the right.
Below the divider, the Generate button triggers `pipeline.run()` inside a spinner.

The auto eye separation checkbox is a UX decision: most users don't know what
"eye separation in pixels" means, so defaulting to `image_width // 8` (the
renderer's own default) is the right out-of-box experience. Power users can
uncheck and dial it manually.

Results show the stereogram image, a download button, and four `st.metric` cards
for the parameters actually used. The input depth map is in an `st.expander` so
the before/after comparison is one click away without cluttering the main view.

Errors from `state["errors"]` surface as `st.warning` inside their own expander —
non-fatal (texture fallback, etc.) don't block the result display.

## Key Design Decisions

**Decision:** No agent files in this project.
**Why:** There are no LLM calls. The pipeline is a single deterministic function.
Adding an agent wrapper would be pure ceremony.

**Decision:** `importlib` over adding `apps/` to `sys.path`.
**Why:** Adding a directory to `sys.path` globally can cause import shadowing in
complex environments. Loading by file path is explicit and scoped.

**Decision:** Temp dir lifetime tied to `pipeline.run()` via `with` block.
**Why:** Guarantees cleanup even if the renderer raises an exception. No temp file
accumulation on the server after repeated runs.

**Decision:** PHI stub on a pure image pipeline.
**Why:** CLAUDE.md requires it on every project. The habit matters more than
whether this specific pipeline could ever encounter PHI.

## How to Explain This in an Interview

**Q: Walk me through the architecture.**
A: "It's a Streamlit UI over a pure-NumPy stereogram renderer. The user uploads a
depth map — greyscale, white-is-near convention — and the renderer shifts each
pixel row left or right by an amount proportional to depth. That shift creates the
parallax that fools the visual cortex into seeing 3D. The UI wraps it in guardrails
and a download button."

**Q: Why does pipeline.py use importlib instead of a normal import?**
A: "The renderer lives in `apps/stereogram-renderer/` — the hyphen in the directory
name makes it un-importable as a Python package. `spec_from_file_location` loads
by absolute file path instead, which sidesteps the naming constraint entirely. In a
production system I'd rename the directory or add a proper package structure, but
for a portfolio monorepo this is clean enough."

**Q: How would you scale this for production traffic?**
A: "Three things: first, replace the Python render loop with a Numba JIT or move
it to a compiled extension — the inner loop has a backward data dependency that
prevents vectorisation but is trivially parallelisable across rows with OpenMP.
Second, add S3 output storage and return a presigned URL instead of streaming bytes
through the server. Third, wire up the rate_limit_check stub to a Redis counter to
prevent abuse."
