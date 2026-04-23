"""
pipeline.py — Stereogram converter pipeline.

Thin wrapper around apps/stereogram-renderer/main.py.
No LLM calls. Accepts image bytes, writes to a temp file,
calls the renderer, reads the result back as bytes.

Flow:
    validate_input(data)
        │
        ▼
    write depth map bytes → temp file
    write texture bytes   → temp file (if provided)
        │
        ▼
    renderer.run({depth_map_path, output_path, texture_path, ...})
        │
        ▼
    read output bytes from temp file
        │
        ▼
    sanitize_output(state)
        │
        ▼
    return state  →  state["output_bytes"], state["output"]
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

from guardrails import sanitize_output, validate_input

# ── Renderer import ────────────────────────────────────────────────────────────
# Load apps/stereogram-renderer/main.py by file path to avoid the hyphen in the
# directory name breaking a normal Python import.

_RENDERER_DIR = Path(__file__).parent.parent.parent / "apps" / "stereogram-renderer"
_RENDERER_FILE = _RENDERER_DIR / "main.py"

if not _RENDERER_FILE.exists():
    raise ImportError(
        f"stereogram-renderer not found at {_RENDERER_FILE}. "
        "Make sure apps/stereogram-renderer/main.py exists."
    )

_spec = importlib.util.spec_from_file_location("stereogram_renderer", _RENDERER_FILE)
_renderer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_renderer)
renderer_run = _renderer.run


# ── State helpers ──────────────────────────────────────────────────────────────

def build_initial_state(validated: dict) -> dict:
    return {
        "input": validated,
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run(user_input: dict) -> dict:
    """
    Convert a depth map image to a stereogram.

    Input keys:
        depth_map_bytes (bytes)      — required
        depth_map_name  (str)        — filename, used to preserve extension
        texture_bytes   (bytes|None) — optional
        texture_name    (str|None)   — optional
        eye_separation  (int|None)   — optional, default image_width // 8
        depth_factor    (float|None) — optional, default 0.33

    Returns state dict with:
        output_bytes   (bytes)      — the rendered stereogram PNG
        output_name    (str)        — suggested filename for download
        output         (str)        — human-readable summary
        width, height  (int)
        eye_separation (int)
        depth_factor   (float)
        errors         (list[str])
    """
    validated = validate_input(user_input)
    state = build_initial_state(validated)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── Write depth map to temp file ───────────────────────────────────────
        state["pipeline_step"] += 1
        depth_name = validated.get("depth_map_name", "depth.png")
        depth_ext = Path(depth_name).suffix or ".png"
        depth_path = tmp_path / f"depth_map{depth_ext}"
        depth_path.write_bytes(validated["depth_map_bytes"])

        # ── Write texture to temp file (if provided) ───────────────────────────
        state["pipeline_step"] += 1
        texture_path = None
        texture_bytes = validated.get("texture_bytes")
        if texture_bytes:
            tex_name = validated.get("texture_name", "texture.png")
            tex_ext = Path(tex_name).suffix or ".png"
            tex_path = tmp_path / f"texture{tex_ext}"
            tex_path.write_bytes(texture_bytes)
            texture_path = str(tex_path)

        # ── Call renderer ──────────────────────────────────────────────────────
        state["pipeline_step"] += 1
        output_path = tmp_path / "stereogram.png"

        renderer_input = {
            "depth_map_path": str(depth_path),
            "output_path":    str(output_path),
            "texture_path":   texture_path,
            "output_format":  "PNG",
        }
        if validated.get("eye_separation"):
            renderer_input["eye_separation"] = validated["eye_separation"]
        if validated.get("depth_factor"):
            renderer_input["depth_factor"] = validated["depth_factor"]

        renderer_result = renderer_run(renderer_input)

        if renderer_result.get("errors"):
            state["errors"].extend(renderer_result["errors"])
        if not renderer_result.get("output_path"):
            state["errors"].append("Renderer failed to produce output.")
            state["output_bytes"] = None
            state["output"] = "Rendering failed."
            return sanitize_output(state)

        # Propagate any renderer warnings
        for e in renderer_result.get("errors", []):
            if e not in state["errors"]:
                state["errors"].append(e)

        # ── Read output bytes ──────────────────────────────────────────────────
        state["pipeline_step"] += 1
        state["output_bytes"] = output_path.read_bytes()

    # ── Populate result metadata ───────────────────────────────────────────────
    stem = Path(depth_name).stem
    state["output_name"]    = f"{stem}_stereogram.png"
    state["width"]          = renderer_result["width"]
    state["height"]         = renderer_result["height"]
    state["eye_separation"] = renderer_result["eye_separation"]
    state["depth_factor"]   = renderer_result["depth_factor"]
    state["output"]         = (
        f"Stereogram generated: {renderer_result['width']}×{renderer_result['height']} px, "
        f"eye_separation={renderer_result['eye_separation']}, "
        f"depth_factor={renderer_result['depth_factor']}"
    )

    return sanitize_output(state)


# ── CLI / self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="stereogram-converter pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate guardrails wiring only; no image processing")
    parser.add_argument("--depth-map", type=str,
                        help="Path to a real depth map image for a full test run")
    parser.add_argument("--texture",   type=str, default=None,
                        help="Path to a texture image (optional)")
    parser.add_argument("--output",    type=str, default="test_stereogram.png",
                        help="Where to save the output (default: test_stereogram.png)")
    args = parser.parse_args()

    if args.dry_run:
        # Confirm guardrails and renderer import without processing any image
        print("[pipeline] Dry run — checking imports and guardrails wiring...")
        try:
            from guardrails import validate_input, sanitize_output, rate_limit_check
            print("  ✓ guardrails imported")
        except ImportError as e:
            print(f"  ✗ guardrails import failed: {e}")

        try:
            _ = renderer_run  # already imported at module load
            print("  ✓ stereogram-renderer imported")
        except Exception as e:
            print(f"  ✗ renderer import failed: {e}")

        test_data = {"depth_map_bytes": b"\x00" * 100, "depth_map_name": "test.png"}
        try:
            validate_input(test_data)
            print("  ✓ validate_input works on minimal input")
        except Exception as e:
            print(f"  ✗ validate_input failed: {e}")

        state = {"errors": []}
        sanitize_output(state)
        print("  ✓ sanitize_output works on empty state")
        print("\n[pipeline] Dry run complete. All wiring OK.")

    elif args.depth_map:
        print(f"[pipeline] Full run: {args.depth_map}")
        depth_path = Path(args.depth_map)
        if not depth_path.exists():
            print(f"Error: depth map not found at {args.depth_map}")
            sys.exit(1)

        user_input = {
            "depth_map_bytes": depth_path.read_bytes(),
            "depth_map_name":  depth_path.name,
        }
        if args.texture:
            tex_path = Path(args.texture)
            user_input["texture_bytes"] = tex_path.read_bytes()
            user_input["texture_name"]  = tex_path.name

        result = run(user_input)

        if result.get("output_bytes"):
            Path(args.output).write_bytes(result["output_bytes"])
            print(f"  Saved: {args.output}")
            print(f"  Size:           {result['width']} × {result['height']} px")
            print(f"  Eye separation: {result['eye_separation']} px")
            print(f"  Depth factor:   {result['depth_factor']}")
        print(f"  Errors: {result['errors'] or 'none'}")

    else:
        print("Usage:")
        print("  python pipeline.py --dry-run")
        print("  python pipeline.py --depth-map depth.png [--texture tex.avif] [--output out.png]")
