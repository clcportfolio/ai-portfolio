# Stereogram Converter — Technical Overview

## Project Goal

A no-LLM `projects/` entry that wraps `apps/stereogram-renderer/` with a Streamlit UI
and guardrails. Demonstrates the stepwise build pattern from CLAUDE.md: pure algorithm
built and validated in `apps/` first, then surfaced as a demoable project.

## Architecture

```
Upload (Streamlit)
      │
      ▼
guardrails.validate_input()
  - image size ≤ 10 MB
  - extension in allowed set
  - depth_factor in [0.05, 0.90]
      │
      ▼
pipeline.run()
  ├─ write depth map bytes → NamedTemporaryFile
  ├─ write texture bytes   → NamedTemporaryFile (if provided)
  ├─ apps/stereogram-renderer/main.py → render()
  └─ read output PNG bytes from temp file
      │
      ▼
guardrails.sanitize_output()
  - PHI pattern scan on error messages (stub)
      │
      ▼
Streamlit: display image + download button
```

## Renderer Algorithm (SIRTS)

Single Image Random Texture Stereogram. For each row, left to right:

```
shift = round((depth[y,x] / 255) × depth_factor × eye_separation)
src   = x - eye_separation + shift

if src >= 0:  result[y,x] = result[y,src]    # parallax copy
else:         result[y,x] = texture[y%Th, x%Tw]  # seed from tile
```

`depth_factor < 1.0` guarantees `src < x` always, so the backward dependency
is satisfied by a single left-to-right sweep. Pre-tiling the texture to canvas
size with `np.tile` gives O(1) lookup per pixel.

## Key Design Decisions

**No LLM, still in `projects/`.**
This project has no agent files, but it has `pipeline.py`, `guardrails.py`, `app.py`,
and docs — the full project contract. It validates the fourth row of the three-folder
rule: "Streamlit UI wrapping an apps/ module, no LLM → projects/."

**importlib over sys.path for renderer import.**
`apps/stereogram-renderer/` has a hyphen in the directory name, which breaks normal
Python imports. `importlib.util.spec_from_file_location` loads the module by file path,
which is unambiguous and works regardless of CWD.

**Bytes in, bytes out at the pipeline boundary.**
Streamlit file uploaders return `BytesIO`-like objects. Keeping the pipeline interface
as `bytes` keeps `pipeline.py` decoupled from Streamlit — it can be called from a CLI,
a test, or another pipeline equally well. Temp files are an implementation detail inside
`pipeline.run()`, invisible to callers.

**PHI stub on an image pipeline.**
There is no text output in this pipeline, but `sanitize_output` scans error message
strings for PHI patterns. This is intentional: it establishes the habit and the
pattern for every future project, including ones that do produce text output.

## Guardrails

`validate_input` checks:
- `depth_map_bytes` present and `isinstance(bytes)`
- Both images ≤ 10 MB
- File extensions in allowed set (`.png`, `.jpg`, `.avif`, etc.)
- `depth_factor` in `[0.05, 0.90]` if provided

`sanitize_output` scans `state["errors"]` for SSN, MRN, and NPI patterns.
`rate_limit_check` is a stub returning `True`.

## Deployment

**Local:** `streamlit run app.py` from `projects/stereogram-converter/`

**Streamlit Community Cloud:**
1. Push repo to GitHub
2. Connect at share.streamlit.io
3. Set main file to `projects/stereogram-converter/app.py`
4. No secrets needed (no API keys)

## Tradeoffs & Known Limitations

- **Performance:** The inner render loop is pure Python; large images (>1024px wide)
  are slow. A Numba JIT or Cython extension would fix this but adds a compilation
  dependency. Acceptable for a portfolio demo.
- **AVIF support:** Requires `libavif` system library. Falls back gracefully with a
  clear error message if missing.
- **No persistence:** Output is not stored; users must download immediately.
  An S3 upload step would be straightforward to add via `boto3`.
