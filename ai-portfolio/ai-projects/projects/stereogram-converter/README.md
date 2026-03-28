# stereogram-converter

Upload a greyscale depth map and download a Magic Eye stereogram. No AI, no API keys.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What you'll see

A two-column UI: upload your depth map on the left, adjust eye separation and depth
factor on the right. Click **Generate Stereogram**, and the result appears below with
a download button. The input depth map is shown in an expander for side-by-side
comparison.

## How it works

```
Upload depth map (+ optional texture)
          │
          ▼
  validate_input()         ← guardrails.py
          │
          ▼
  write to temp file
          │
          ▼
  apps/stereogram-renderer/main.py → render()
          │
          ▼
  read output bytes
          │
          ▼
  sanitize_output()        ← guardrails.py
          │
          ▼
  Display + download button
```

## Depth map format

| Pixel value | Meaning |
|---|---|
| White (255) | Close to viewer |
| Black (0)   | Far from viewer |
| Mid-grey    | Intermediate depth |

The example depth map in this repo (`docs/example_depth.png`) is a greyscale render
of a 3D figure against a black background.

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| Eye separation | image width ÷ 8 | Wider = easier to view; narrower = denser pattern |
| Depth factor | 0.33 | Higher = stronger 3D pop; above 0.5 can cause eyestrain |

## Tech stack

- NumPy + Pillow — SIRTS algorithm
- Streamlit — UI
- No LLM / no API keys required

## Part of a larger system

This project is the "direct upload" frontend for `apps/stereogram-renderer/`.
The same renderer is used by `projects/stereogram-pipeline/` which adds a text
prompt → AI image generation → depth map step before rendering.
