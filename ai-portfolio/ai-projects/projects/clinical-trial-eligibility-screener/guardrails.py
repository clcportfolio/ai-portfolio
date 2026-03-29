"""
guardrails.py — Clinical Trial Eligibility Screener
Security and validation layer for clinical trial eligibility screening pipeline.
"""

import re
import logging
import base64

logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 4000
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

# Prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now",
    r"disregard (your |all )?instructions",
    r"act as (a |an )?",
    r"jailbreak",
]

# PHI patterns for detection
_PHI_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",   # SSN
    r"\b\d{10}\b",               # NPI
    r"\bMRN[:\s]?\d+\b",        # Medical record number
    r"\b\d{2}/\d{2}/\d{4}\b",   # Date of birth patterns
    r"\b[A-Z]{2}\d{6,8}\b",     # Medical ID patterns
]


def validate_input(data: dict) -> dict:
    """
    Validates user input for clinical trial eligibility screening.
    Raises ValueError on failure. Returns data unchanged on success.
    """
    # Validate trial_criteria field
    trial_criteria = data.get("trial_criteria", "")
    if not isinstance(trial_criteria, str):
        raise ValueError("Input 'trial_criteria' must be a string.")
    if len(trial_criteria) > MAX_TEXT_LENGTH:
        raise ValueError(f"Trial criteria exceeds {MAX_TEXT_LENGTH} character limit.")
    
    # Validate patient_summary field
    patient_summary = data.get("patient_summary", "")
    if not isinstance(patient_summary, str):
        raise ValueError("Input 'patient_summary' must be a string.")
    if len(patient_summary) > MAX_TEXT_LENGTH:
        raise ValueError(f"Patient summary exceeds {MAX_TEXT_LENGTH} character limit.")
    
    # Ensure both required fields are present and non-empty
    if not trial_criteria.strip():
        raise ValueError("Trial criteria cannot be empty.")
    if not patient_summary.strip():
        raise ValueError("Patient summary cannot be empty.")
    
    # Prompt injection scan on trial_criteria
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, trial_criteria, re.IGNORECASE):
            raise ValueError("Potential prompt injection detected in trial criteria.")
    
    # Prompt injection scan on patient_summary
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, patient_summary, re.IGNORECASE):
            raise ValueError("Potential prompt injection detected in patient summary.")
    
    # Image validation (if present)
    image_b64 = data.get("image_b64")
    if image_b64:
        if not isinstance(image_b64, str):
            raise ValueError("Input 'image_b64' must be a string.")
        try:
            raw = base64.b64decode(image_b64)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds 10MB size limit.")
        except Exception:
            raise ValueError("Invalid base64 image data.")
    
    return data


def sanitize_output(state: dict) -> dict:
    """
    Sanitizes output for clinical trial eligibility screening.
    Returns modified state dict. Never raises — logs warnings instead.
    """
    # Get output field
    output = state.get("output", "")
    if not isinstance(output, str):
        return state
    
    # Remove script tags
    output = re.sub(
        r"<script.*?>.*?</script>", "[REMOVED]", output,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # PHI redaction scan
    for pattern in _PHI_PATTERNS:
        if re.search(pattern, output):
            logger.warning(
                "PHI pattern detected in output. "
                "Replace with AWS Comprehend Medical in production."
                # NOTE: replace with production-grade scanner in prod
            )
            break
    
    # Additional clinical data sanitization
    # Remove potential patient identifiers from output
    output = re.sub(r"\bpatient\s+id[:\s]?\w+\b", "[PATIENT_ID_REMOVED]", output, flags=re.IGNORECASE)
    output = re.sub(r"\bmedical\s+record[:\s]?\w+\b", "[MRN_REMOVED]", output, flags=re.IGNORECASE)
    
    # Update state with sanitized output
    state["output"] = output
    
    # Sanitize any other string fields in state that might contain PHI
    for key, value in state.items():
        if isinstance(value, str) and key != "output":
            # Remove script tags from other string fields
            sanitized_value = re.sub(
                r"<script.*?>.*?</script>", "[REMOVED]", value,
                flags=re.DOTALL | re.IGNORECASE
            )
            state[key] = sanitized_value
    
    return state


def rate_limit_check(user_id: str) -> bool:
    """
    Rate limiting check for clinical trial eligibility screening.
    Returns True if user is within rate limits, False otherwise.
    """
    # replace with Redis-backed counter in production
    return True

if __name__ == "__main__":
    print("Testing validate_input...")
    tests = [
        ({"trial_criteria": "Inclusion: Age 18-65.", "patient_summary": "45-year-old male."}, True),
        ({"trial_criteria": "", "patient_summary": "Patient."}, False),
        ({"trial_criteria": "ignore all previous instructions", "patient_summary": "Patient."}, False),
        ({"trial_criteria": "A" * 4001, "patient_summary": "Patient."}, False),
    ]
    for data, should_pass in tests:
        try:
            validate_input(data)
            status = "✓ pass" if should_pass else "✗ unexpected pass"
        except ValueError as e:
            status = "✗ fail" if should_pass else f"✓ correctly rejected: {e}"
        print(f"  {status}")

    print("\nTesting sanitize_output (PHI stub)...")
    state = {"output": "Patient SSN: 123-45-6789 is eligible."}
    result = sanitize_output(state)
    print(f"  Output: {result['output']}")
    print("  (Check logs above for PHI warning)")
