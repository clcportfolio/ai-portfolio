# stereogram-renderer

Pure-algorithm depth map → stereogram converter. No LLM. Zero API keys.
A building block for `projects/stereogram-converter/` and `projects/stereogram-pipeline/`.

## Run it (CLI)

```bash
pip install -r requirements.txt
python main.py depth_map.png output.png
python main.py depth_map.png output.png texture.avif   # with texture tile
```

## Local test files (not in repo)

Test data lives in `tmp_test_data/` which is gitignored. To run the CLI locally you need:

- A depth map image (grayscale PNG/JPG — white = near, black = far)
- Optional: a texture tile (any image — AVIF recommended)

Create the folder and drop in your own files:
```
apps/stereogram-renderer/tmp_test_data/
├── inputs/      ← your depth map(s) here
├── outputs/     ← generated stereograms written here
└── textures/    ← optional texture tiles here
```

## Use it as a module

```python
from main import run

result = run({
    "depth_map_path": "depth.png",
    "output_path":    "stereogram.png",
    "texture_path":   "texture.avif",   # optional — random noise used if omitted
    "eye_separation": 100,              # optional — default: image_width // 8
    "depth_factor":   0.33,             # optional — controls depth intensity (0.05–0.90)
})
# result["output_path"], result["width"], result["height"], result["errors"]
```

## Depth map format

| Pixel value | Meaning |
|---|---|
| Black (0) | Far from viewer (background) |
| White (255) | Close to viewer (foreground) |
| Mid-grey | Intermediate depth |

Colour images are automatically converted to luminance. The resolution of the
input image is maintained exactly in the output.

## How it works

```
depth_map.png + texture tile
        │
        ▼
  For each row, left to right:
    shift = round((depth / 255) × depth_factor × eye_separation)
    src   = x - eye_separation + shift

    if src >= 0 → copy pixel from result[y, src]   (parallax)
    else        → seed from texture tile            (pattern origin)
        │
        ▼
  stereogram.png
```

The left `eye_separation` pixels of each row are always seeded from the texture
(their `src` is out of bounds). Every pixel to the right copies from a position
that is slightly closer or further left depending on depth, which fools the visual
cortex into perceiving a 3D surface when eyes are de-focused.

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `eye_separation` | `width // 8` | Wider = easier to view; narrower = denser pattern |
| `depth_factor` | `0.33` | Higher = stronger 3D pop; above 0.5 can cause eyestrain |

## Image format support

**Input:** `.png` `.jpg` `.jpeg` `.bmp` `.webp` `.tiff` `.tif` `.avif`

**Output:** `.png` `.jpg` `.jpeg` `.bmp` `.webp` `.tiff` `.tif`

> **AVIF note:** Requires Pillow ≥ 10.0 with `libavif` installed.
> On macOS: `brew install libavif` then `pip install Pillow --upgrade`.
> On Ubuntu: `apt-get install libavif-dev` then reinstall Pillow.

## Viewing tip

Hold the image at arm's length. Relax your focus as if looking through the screen
at something far behind it ("magic eye" technique). The 3D image should snap into
view within a few seconds. Wider eye separation is easier for first-time viewers.

## What uses this

| Consumer | How |
|---|---|
| `projects/stereogram-converter/` | Streamlit UI — user uploads depth map, downloads stereogram |
| `projects/stereogram-pipeline/` | AI pipeline — text prompt → depth map → this renderer |
