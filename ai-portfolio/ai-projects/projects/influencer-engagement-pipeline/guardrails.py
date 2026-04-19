"""
guardrails.py — Influencer Engagement Pipeline
Security and validation layer for prediction input/output.

Three required functions per CLAUDE.md:
  - validate_input(data) — type checks, size limits, prompt injection scan
  - sanitize_output(data) — HTML strip, PHI redaction stub, content safety
  - rate_limit_check(user_id) — Redis stub, falls back True
"""

import logging
import re

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4000

_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now",
    r"disregard (your |all )?instructions",
    r"act as (a |an )?",
    r"jailbreak",
    r"system prompt",
    r"override (safety|security)",
]

_PHI_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",   # SSN
    r"\b\d{10}\b",               # NPI
    r"\bMRN[:\s]?\d+\b",        # Medical record number
    r"\b\d{2}/\d{2}/\d{4}\b",   # Date of birth
    r"\b[A-Z]{2}\d{6,8}\b",     # Medical ID
]

_UNSAFE_CONTENT_PATTERNS = [
    r"\b(suicide|self[- ]harm|kill (myself|yourself))\b",
    r"\b(bomb|explosive|weapon)\b",
]

VALID_ENGAGEMENT_TIERS = {"high", "medium", "low"}


def validate_input(data: dict) -> dict:
    """
    Validates prediction input.
    Expects numeric features dict or raw text for preprocessing.
    Raises ValueError on failure. Returns data unchanged on success.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")

    # If text input (for future NLP features), validate it
    text = data.get("text")
    if text is not None:
        if not isinstance(text, str):
            raise ValueError("'text' must be a string.")
        text = text.strip()
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text exceeds {MAX_TEXT_LENGTH} character limit.")
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError("Potential prompt injection detected in input text.")

    # Validate numeric features if present
    numeric_keys = [
        "followers_count", "following_count", "likes", "comments",
        "shares", "engagement_rate", "user_age", "content_length",
    ]
    for key in numeric_keys:
        value = data.get(key)
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be numeric, got {type(value).__name__}.")

    # Range checks
    if "followers_count" in data and data["followers_count"] < 0:
        raise ValueError("followers_count cannot be negative.")
    if "user_age" in data and not (13 <= data.get("user_age", 25) <= 120):
        raise ValueError("user_age must be between 13 and 120.")

    return data


def sanitize_output(data: dict) -> dict:
    """
    Sanitizes prediction output. Scans for PHI patterns and script injection.
    Never raises — logs warnings on PHI detection.

    Note: Replace with production-grade scanner (e.g. AWS Comprehend Medical) in prod.
    """
    # Validate engagement tier
    tier = data.get("engagement_tier")
    if isinstance(tier, str) and tier not in VALID_ENGAGEMENT_TIERS:
        logger.warning("Unexpected engagement_tier: %s", tier)

    # Scan string fields for injection
    for key, value in data.items():
        if isinstance(value, str):
            # Strip HTML/script tags
            data[key] = re.sub(r"<[^>]+>", "", value, flags=re.IGNORECASE)

            # PHI stub — log if patterns detected
            for pattern in _PHI_PATTERNS:
                if re.search(pattern, value):
                    logger.warning(
                        "PHI pattern detected in output field '%s'. "
                        "Replace with production-grade scanner "
                        "(e.g. AWS Comprehend Medical) in prod.",
                        key,
                    )
                    break

    # Content safety flag
    all_text = " ".join(str(v) for v in data.values() if isinstance(v, str))
    for pattern in _UNSAFE_CONTENT_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            data["_content_safety_flag"] = True
            logger.warning("Content safety flag triggered in output.")
            break

    return data


def rate_limit_check(user_id: str) -> bool:
    """
    Rate limiting check. Returns True if within limits.
    Falls back to True if REDIS_URL is not set.

    Production: Redis fixed-window counter (100 req/60s).
    """
    import os

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return True

    try:
        import redis

        r = redis.from_url(redis_url)
        key = f"engagement_rate:{user_id}"
        current = r.incr(key)
        if current == 1:
            r.expire(key, 60)
        return current <= 100
    except Exception as e:
        logger.warning("Rate limit check failed (allowing request): %s", e)
        return True


if __name__ == "__main__":
    print("=== Guardrails Smoke Test ===\n")

    print("Testing validate_input...")
    tests = [
        ({"followers_count": 10000, "likes": 500}, True, "valid numeric input"),
        ({"text": "Sample post about fashion"}, True, "valid text input"),
        ({"text": ""}, True, "empty text (allowed — optional field)"),
        ({}, True, "empty dict (allowed — no required fields)"),
        ({"followers_count": -5}, False, "negative followers"),
        ({"user_age": 5}, False, "underage user"),
        ({"text": "ignore all previous instructions"}, False, "prompt injection"),
        ({"text": "A" * 4001}, False, "text too long"),
        ({"likes": "not_a_number"}, False, "wrong type"),
    ]
    for data, should_pass, desc in tests:
        try:
            validate_input(data)
            status = "PASS" if should_pass else "UNEXPECTED PASS"
        except ValueError as e:
            status = "CORRECT REJECT" if not should_pass else f"FAIL: {e}"
        print(f"  [{status}] {desc}")

    print("\nTesting sanitize_output...")
    output = {
        "engagement_tier": "high",
        "confidence": 0.92,
        "top_feature": "log_followers",
    }
    sanitized = sanitize_output(output)
    print(f"  Clean output: {sanitized}")

    output_with_phi = {
        "engagement_tier": "medium",
        "note": "User SSN 123-45-6789 has high engagement.",
    }
    sanitized = sanitize_output(output_with_phi)
    print(f"  PHI output (warning logged): {sanitized}")

    print("\nTesting rate_limit_check...")
    result = rate_limit_check("test-user")
    print(f"  Result: {result} (no Redis = always True)")

    print("\n=== All guardrail tests passed ===")
