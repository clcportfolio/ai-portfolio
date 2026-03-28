"""
guardrails.py — Input/output safety middleware for stereogram-converter.

No LLM in this project, but guardrails are still present:
  - Image size and format validation
  - PHI redaction stub (required in every project per CLAUDE.md)
  - Rate limit stub
"""

import logging
import re

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".avif",
}

# PHI patterns — stub scanner, logs a warning if triggered
# NOTE: replace with AWS Comprehend Medical in production
_PHI_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",   # SSN
    r"\bMRN[:\s]?\d+\b",         # Medical record number
    r"\b\d{10}\b",               # NPI
]


def validate_input(data: dict) -> dict:
    """
    Validate pipeline input before processing.

    Expects:
        data["depth_map_bytes"]  (bytes)       — required
        data["depth_map_name"]   (str)          — required, used for format detection
        data["texture_bytes"]    (bytes | None) — optional
        data["texture_name"]     (str | None)   — optional
        data["eye_separation"]   (int | None)   — optional
        data["depth_factor"]     (float | None) — optional

    Raises ValueError on any validation failure.
    """
    # Depth map — required
    depth_bytes = data.get("depth_map_bytes")
    if not depth_bytes:
        raise ValueError("No depth map image provided.")
    if not isinstance(depth_bytes, (bytes, bytearray)):
        raise ValueError("depth_map_bytes must be bytes.")
    if len(depth_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Depth map exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB limit "
            f"({len(depth_bytes) // (1024*1024)} MB received)."
        )

    depth_name = data.get("depth_map_name", "")
    depth_ext = "." + depth_name.rsplit(".", 1)[-1].lower() if "." in depth_name else ""
    if depth_ext and depth_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Depth map format '{depth_ext}' is not supported. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # Texture — optional; validate only if provided
    texture_bytes = data.get("texture_bytes")
    if texture_bytes is not None:
        if not isinstance(texture_bytes, (bytes, bytearray)):
            raise ValueError("texture_bytes must be bytes.")
        if len(texture_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Texture image exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB limit."
            )
        texture_name = data.get("texture_name", "")
        texture_ext = "." + texture_name.rsplit(".", 1)[-1].lower() if "." in texture_name else ""
        if texture_ext and texture_ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Texture format '{texture_ext}' is not supported. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            )

    # Numeric parameters — clamp, don't reject
    eye_sep = data.get("eye_separation")
    if eye_sep is not None and (not isinstance(eye_sep, int) or eye_sep < 1):
        raise ValueError("eye_separation must be a positive integer.")

    depth_factor = data.get("depth_factor")
    if depth_factor is not None:
        if not isinstance(depth_factor, (int, float)):
            raise ValueError("depth_factor must be a number.")
        if not (0.05 <= float(depth_factor) <= 0.90):
            raise ValueError("depth_factor must be between 0.05 and 0.90.")

    return data


def sanitize_output(state: dict) -> dict:
    """
    Post-processing safety check on pipeline output.

    For an image pipeline there is no text output to sanitize, but:
      - Errors list is scanned for accidental PHI leakage in error messages
      - PHI stub is always present per CLAUDE.md convention
    """
    # Scan error messages for PHI patterns (defensive — error strings shouldn't contain PHI,
    # but if a file path or metadata leaked into an error we want to know)
    errors = state.get("errors", [])
    for msg in errors:
        for pattern in _PHI_PATTERNS:
            if re.search(pattern, str(msg)):
                logger.warning(
                    "PHI pattern detected in pipeline error message. "
                    "Review error handling to prevent PHI leakage into logs. "
                    # NOTE: replace with AWS Comprehend Medical in production
                )
                break

    return state


def rate_limit_check(user_id: str) -> bool:
    """
    Stub — always returns True (allow).
    Replace with Redis-backed counter in production.
    """
    return True


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="guardrails smoke test for stereogram-converter")
    parser.add_argument("--dry-run", action="store_true", help="Print test plan only")
    args = parser.parse_args()

    tests = [
        ("valid input",        {"depth_map_bytes": b"\x00" * 100, "depth_map_name": "depth.png"}, None),
        ("empty bytes",        {"depth_map_bytes": b"",            "depth_map_name": "depth.png"}, ValueError),
        ("missing bytes",      {"depth_map_name": "depth.png"},                                    ValueError),
        ("bad extension",      {"depth_map_bytes": b"\x00" * 100, "depth_map_name": "depth.exe"}, ValueError),
        ("oversized image",    {"depth_map_bytes": b"\x00" * (11 * 1024 * 1024), "depth_map_name": "d.png"}, ValueError),
        ("bad depth_factor",   {"depth_map_bytes": b"\x00" * 100, "depth_map_name": "d.png", "depth_factor": 2.0}, ValueError),
    ]

    if args.dry_run:
        print("[guardrails] Dry run — test plan:")
        for name, _, expected in tests:
            print(f"  {name:30s}  expect: {'ValueError' if expected else 'pass'}")
    else:
        print("[guardrails] Running validate_input tests...")
        passed = failed = 0
        for name, data, expected_exc in tests:
            try:
                validate_input(data)
                result = "PASS" if expected_exc is None else "FAIL (no exception raised)"
            except ValueError as e:
                result = "PASS" if expected_exc is ValueError else f"FAIL (unexpected: {e})"
            print(f"  {'✓' if 'PASS' in result else '✗'} {name:30s}  {result}")
            if "PASS" in result:
                passed += 1
            else:
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
        print("\n[guardrails] sanitize_output and rate_limit_check: always pass (stubs).")
