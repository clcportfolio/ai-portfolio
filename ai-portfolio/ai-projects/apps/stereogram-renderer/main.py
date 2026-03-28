"""
stereogram-renderer — Pure algorithm. No LLM.

Converts a depth map image into a single-image random-texture stereogram (SIRTS).

Depth map convention:
  - Black (0)   = far from viewer
  - White (255) = close to viewer
  - Any greyscale image works; colour images are converted to luminance

Algorithm:
  For each row, left to right:
    frac_src = x - eye_separation + (depth / 255) * depth_factor * eye_separation

    The fractional source position is bilinearly interpolated between its two
    neighbouring already-computed pixels, preserving all 256 depth levels as
    smooth blended transitions instead of quantised integer steps.

    if floor(frac_src) >= 0:
        result[y, x] = lerp(result[y, floor(frac_src)],
                            result[y, floor(frac_src)+1], frac)
    else:
        result[y, x] = texture[y%Th, x%Tw]   # seed from texture tile

  Because depth_factor < 1.0, frac_src is always < x, so both neighbours are
  already computed and the left-to-right pass remains correct in a single sweep.

Viewing tip: hold the image at arm's length and relax your focus past the screen
("magic eye" technique) until the 3D image snaps into view.

Usage as a module:
    from main import run
    result = run({
        "depth_map_path": "depth.png",
        "output_path":    "out.png",
        "texture_path":   "texture.avif",   # optional
        "eye_separation": 100,              # optional, default image_width // 8
        "depth_factor":   0.33,             # optional, default 0.33
    })

Usage as a CLI:
    python main.py depth.png out.png [texture.avif]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── Format support ─────────────────────────────────────────────────────────────

# Formats Pillow can read on most installs.
# AVIF requires Pillow >= 9.1 with libavif; see README for install notes.
SUPPORTED_INPUT_FORMATS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".webp", ".tiff", ".tif", ".avif",
}

SUPPORTED_OUTPUT_FORMATS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif",
}


# ── Image I/O ──────────────────────────────────────────────────────────────────

def _load_image(path: str) -> Image.Image:
    """Load an image file, raising clear errors for bad paths or formats."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: '{path}'")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            f"Unsupported input format '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_INPUT_FORMATS)}"
        )
    try:
        img = Image.open(path)
        img.load()  # force decode now so errors surface here, not later
        return img
    except Exception as e:
        raise IOError(f"Could not read '{path}': {e}") from e


def _to_gray(img: Image.Image) -> np.ndarray:
    """Convert any PIL image to a 2-D uint8 array (luminance)."""
    return np.array(img.convert("L"), dtype=np.uint8)


def _to_rgb(img: Image.Image) -> np.ndarray:
    """Convert any PIL image to an H×W×3 uint8 array."""
    return np.array(img.convert("RGB"), dtype=np.uint8)


# ── Texture generation ─────────────────────────────────────────────────────────

def _random_noise_texture(tile_size: int = 128, seed: int = 42) -> np.ndarray:
    """
    Generate a coloured random-noise tile.
    The seed makes output reproducible; change it for a different look.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (tile_size, tile_size, 3), dtype=np.uint8)


def _tile_to_canvas(texture: np.ndarray, H: int, W: int) -> np.ndarray:
    """Repeat texture tile to cover an H×W canvas. Crops to exact size."""
    Th, Tw = texture.shape[:2]
    reps_h = -(-H // Th)   # ceiling division
    reps_w = -(-W // Tw)
    return np.tile(texture, (reps_h, reps_w, 1))[:H, :W]


# ── Core algorithm ─────────────────────────────────────────────────────────────

def render(
    depth_map: np.ndarray,
    texture: np.ndarray,
    eye_separation: int,
    depth_factor: float,
) -> np.ndarray:
    """
    Render a stereogram from a depth map and texture tile.

    Args:
        depth_map:      H×W uint8 array, 0 = far, 255 = near.
        texture:        Th×Tw×3 uint8 tile. Tiled to fill canvas automatically.
        eye_separation: Horizontal pixel distance between eyes (controls 3D width).
        depth_factor:   Max depth shift as a fraction of eye_separation (0.05–0.90).
                        Higher = more pronounced 3D effect. 0.33 is a safe default.

    Returns:
        H×W×3 uint8 stereogram array.
    """
    H, W = depth_map.shape
    tiled = _tile_to_canvas(texture, H, W)

    result = np.zeros((H, W, 3), dtype=np.uint8)
    # Keep max_shift as float — no rounding here preserves sub-pixel precision
    max_shift = depth_factor * eye_separation

    for y in range(H):
        # Full float precision — 256 depth levels stay as 256 distinct shifts
        float_shifts = depth_map[y].astype(np.float32) / 255.0 * max_shift

        row = result[y]      # view into result — writes propagate automatically
        tex_row = tiled[y]

        for x in range(W):
            frac_src = x - eye_separation + float_shifts[x]
            src_lo = int(frac_src)          # floor
            alpha = frac_src - src_lo       # blend weight toward src_hi

            if src_lo >= 0:
                src_hi = src_lo + 1
                lo = row[src_lo].astype(np.float32)
                # src_hi is guaranteed < x when depth_factor < 1.0;
                # clamp to src_lo on the rare edge where they coincide
                hi = row[src_hi].astype(np.float32) if src_hi < x else lo
                row[x] = (lo * (1.0 - alpha) + hi * alpha).astype(np.uint8)
            else:
                row[x] = tex_row[x]

    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def run(input: dict) -> dict:
    """
    Convert a depth map image into a stereogram.

    Input keys:
        depth_map_path (str)   — required. Path to depth map image.
        output_path    (str)   — required. Where to save the stereogram.
        texture_path   (str)   — optional. Texture tile image. Random noise if omitted.
        eye_separation (int)   — optional. Default: image_width // 8.
        depth_factor   (float) — optional. 0.05–0.90. Default: 0.33.
        output_format  (str)   — optional. 'PNG', 'JPEG', etc.
                                  Inferred from output_path extension if omitted.

    Returns dict:
        output_path    (str)
        width          (int)
        height         (int)
        eye_separation (int)   — the value actually used
        depth_factor   (float) — the value actually used
        errors         (list[str]) — non-fatal warnings (e.g. texture fallback)
    """
    errors: list[str] = []

    # ── Validate required inputs ───────────────────────────────────────────────
    depth_map_path = input.get("depth_map_path")
    output_path    = input.get("output_path")

    if not depth_map_path:
        return {"errors": ["'depth_map_path' is required."], "output_path": None}
    if not output_path:
        return {"errors": ["'output_path' is required."], "output_path": None}

    # ── Load depth map ─────────────────────────────────────────────────────────
    try:
        depth_img = _load_image(depth_map_path)
    except (FileNotFoundError, ValueError, IOError) as e:
        return {"errors": [str(e)], "output_path": None}

    depth_array = _to_gray(depth_img)
    H, W = depth_array.shape

    # ── Load or generate texture ───────────────────────────────────────────────
    texture_path = input.get("texture_path")
    if texture_path:
        try:
            texture = _to_rgb(_load_image(texture_path))
        except (FileNotFoundError, ValueError, IOError) as e:
            errors.append(f"Texture load failed — using random noise. Reason: {e}")
            texture = _random_noise_texture()
    else:
        texture = _random_noise_texture()

    # ── Parameters ────────────────────────────────────────────────────────────
    eye_separation = int(input.get("eye_separation") or W // 8)
    depth_factor   = float(input.get("depth_factor") or 0.33)
    depth_factor   = max(0.05, min(0.90, depth_factor))  # clamp to safe range

    if eye_separation < 10:
        errors.append(f"eye_separation={eye_separation} is very small; result may look flat.")
    if eye_separation >= W:
        return {"errors": [f"eye_separation ({eye_separation}) must be less than image width ({W})."], "output_path": None}

    # ── Render ────────────────────────────────────────────────────────────────
    stereo_array = render(depth_array, texture, eye_separation, depth_factor)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    if out_path.suffix.lower() not in SUPPORTED_OUTPUT_FORMATS:
        errors.append(
            f"Output extension '{out_path.suffix}' may not be supported — saving as PNG instead."
        )
        out_path = out_path.with_suffix(".png")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = input.get("output_format")
    if not fmt:
        fmt = out_path.suffix.lstrip(".").upper() or "PNG"
    if fmt.upper() == "JPG":
        fmt = "JPEG"

    out_img = Image.fromarray(stereo_array, mode="RGB")
    try:
        out_img.save(str(out_path), format=fmt.upper())
    except Exception as e:
        return {"errors": [f"Failed to save output image: {e}"] + errors, "output_path": None}

    return {
        "output_path": str(out_path),
        "width":        W,
        "height":       H,
        "eye_separation": eye_separation,
        "depth_factor":   depth_factor,
        "errors":         errors,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <depth_map> <output_path> [texture_path]")
        print("  depth_map    — grayscale depth image (black=far, white=near)")
        print("  output_path  — where to save the stereogram (PNG recommended)")
        print("  texture_path — optional texture tile (.png, .jpg, .avif, …)")
        sys.exit(1)

    result = run({
        "depth_map_path": sys.argv[1],
        "output_path":    sys.argv[2],
        "texture_path":   sys.argv[3] if len(sys.argv) > 3 else None,
    })
    print(json.dumps(result, indent=2))

    if result.get("output_path"):
        print(f"\nStereogram saved to: {result['output_path']}")
        print(f"  Size:           {result['width']} × {result['height']} px")
        print(f"  Eye separation: {result['eye_separation']} px")
        print(f"  Depth factor:   {result['depth_factor']}")
    else:
        print("\nFailed. Errors above.")
        sys.exit(1)
