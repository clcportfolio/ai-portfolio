# SKILLS.md — guardrail_engineer agent
## Domain: Guardrails Generation for LangChain Projects

Reference for generating production-quality `guardrails.py` files.
Every guardrails file this agent produces must pass the checklist below.

---

## 1. The Three Required Functions — Exact Signatures

```python
def validate_input(data: dict) -> dict:
    """Raises ValueError on failure. Returns data unchanged on success."""

def sanitize_output(state: dict) -> dict:
    """Returns modified state dict. Never raises — logs warnings instead."""

def rate_limit_check(user_id: str) -> bool:
    """Stub. Returns True. Replace with Redis-backed counter in production."""
```

Never change these signatures. `pipeline.py` always calls them this way.

---

## 2. validate_input Checklist

Every generated `validate_input` must include ALL of the following:

### Text input (if the project uses text)
```python
MAX_TEXT_LENGTH = 4000

text = data.get("text", "")
if not isinstance(text, str):
    raise ValueError("Input 'text' must be a string.")
if len(text) > MAX_TEXT_LENGTH:
    raise ValueError(f"Input exceeds {MAX_TEXT_LENGTH} character limit.")
```

### Image input (if the project uses images)
```python
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

image_b64 = data.get("image_b64")
if image_b64:
    raw = base64.b64decode(image_b64)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds 10MB size limit.")
```

### Prompt injection scan (ALL projects, always)
```python
_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now",
    r"disregard (your |all )?instructions",
    r"act as (a |an )?",
    r"jailbreak",
]

for pattern in _INJECTION_PATTERNS:
    if re.search(pattern, text, re.IGNORECASE):
        raise ValueError(f"Potential prompt injection detected.")
```

---

## 3. sanitize_output Checklist

### Script tag removal (ALL projects)
```python
output = re.sub(
    r"<script.*?>.*?</script>", "[REMOVED]", output,
    flags=re.DOTALL | re.IGNORECASE
)
```

### PHI redaction stub (ALL projects — non-negotiable)
```python
_PHI_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",   # SSN
    r"\b\d{10}\b",               # NPI
    r"\bMRN[:\s]?\d+\b",        # Medical record number
]

for pattern in _PHI_PATTERNS:
    if re.search(pattern, output):
        logger.warning(
            "PHI pattern detected in output. "
            "Replace with AWS Comprehend Medical in production."
            # NOTE: replace with production-grade scanner in prod
        )
        break
```

**This stub is required in every project, even non-clinical ones.**
It signals production and compliance instincts. If an interviewer asks why it's
in a toy-safety-checker, the answer is: "Every project in this portfolio is
architected as if it could handle sensitive data — it's easier to remove a safety
check than to add one retroactively."

### rate_limit_check stub (ALL projects)
```python
def rate_limit_check(user_id: str) -> bool:
    # replace with Redis-backed counter in production
    return True
```

---

## 4. Inferring Input Shape from pipeline.py

The LLM is given `pipeline.py` as context. Use it to determine:
- What keys does `build_initial_state()` put in the state dict?
- What does `validate_input(user_input)` receive — text, image, both?
- Are there any domain-specific fields (e.g. `patient_id`, `form_type`) that need type-checking?

Generate type checks specific to those fields. Don't just check `text` if the
pipeline also has `image_b64`, `document_type`, etc.

---

## 5. Fallback Template

If the LLM call fails, `_fallback_guardrails()` returns a minimal complete template.
The fallback always compiles and runs correctly. It covers text-only pipelines.
For image pipelines the fallback may need manual adjustment — note this in `errors`.

---

## 6. Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| `state["errors"] = [msg]` in sanitize_output | Never reassign errors; only log warnings |
| Raising in sanitize_output | sanitize_output never raises — it logs and continues |
| Missing PHI stub | Always present, even for non-clinical projects |
| Injection scan only on first field | Scan all user-controlled string fields |
| `rate_limit_check` returns `False` | Always returns `True` (stub) |
