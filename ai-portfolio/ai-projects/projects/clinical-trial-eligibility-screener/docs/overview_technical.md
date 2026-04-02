## Project Goal

Automate clinical trial eligibility screening by evaluating patient summaries against trial criteria and providing clear verdicts with reasoning to help coordinators make informed decisions. The system extracts structured criteria from raw trial text, evaluates each criterion individually against patient data, and synthesizes a final eligibility verdict with plain-English explanations.

## Architecture Overview

```
Input (trial_criteria, patient_summary)
    ↓
validate_input() → check injection patterns, PHI, length limits
    ↓
criteria_agent → extract/structure inclusion/exclusion criteria
    ↓
evaluation_agent → evaluate patient against each criterion
    ↓
verdict_agent → synthesize final eligibility verdict
    ↓
sanitize_output() → redact PHI, validate output structure
    ↓
Output (eligibility verdict + reasoning)
```

## Agent Roles

**criteria_agent**: Reads raw `trial_criteria` text from state["input"], uses LangChain's `LLMChain` pattern with structured output parsing to extract individual inclusion/exclusion criteria into standardized format. Writes structured criteria list to state["structured_criteria"] for downstream evaluation.

**evaluation_agent**: Reads state["structured_criteria"] and state["input"]["patient_summary"], implements LangChain's `MapReduceDocumentsChain` pattern to evaluate patient against each criterion individually. Generates detailed reasoning for each criterion match/mismatch and writes evaluation results to state["individual_evaluations"].

**verdict_agent**: Reads state["individual_evaluations"] containing all criterion assessments, uses LangChain's `ConversationChain` with custom prompt template to synthesize final eligibility verdict. Writes comprehensive verdict with plain-English explanation to state["output"]["verdict"] and state["output"]["reasoning"].

## LLM Choices & Rationale

| Agent | Model | Temperature | Rationale |
|-------|-------|-------------|-----------|
| criteria_agent | claude-3-haiku-20240307 | 0.1 | Fast, cost-effective for structured extraction; low temperature ensures consistent parsing of medical criteria |
| evaluation_agent | claude-3-sonnet-20240229 | 0.2 | Balanced reasoning capability for complex medical evaluations; slightly higher temperature for nuanced clinical judgment |
| verdict_agent | claude-3-sonnet-20240229 | 0.3 | Strong synthesis capabilities for final decision-making; moderate temperature for clear, varied explanations |

## Guardrails Design

`validate_input()` checks for prompt injection patterns (`_INJECTION_PATTERNS`), enforces `MAX_TEXT_LENGTH` limits on trial criteria and patient summaries, and validates required field types. `sanitize_output()` scans for PHI patterns (`_PHI_PATTERNS`) including SSNs, NPIs, MRNs, and medical IDs, replacing matches with `[REDACTED]` placeholders. PHI detection uses regex stubs rather than comprehensive medical entity recognition to balance privacy protection with development speed—production systems would integrate specialized medical NLP libraries.

## Langfuse Observability

Each agent execution creates nested traces: `clinical_trial_screening.{agent_name}` with input/output logging, token usage, and latency metrics. Pipeline runs generate parent trace `clinical_trial_screening.pipeline` containing agent spans plus guardrail validation steps. Dashboard shows success rates per agent, average processing times, and PHI redaction frequency. Custom metadata includes criterion count, evaluation complexity scores, and verdict confidence levels.

## Data Flow

User submits trial criteria text and patient summary through Streamlit interface → `validate_input()` security checks → `criteria_agent` extracts structured criteria using `ChatAnthropic` with Pydantic output parser → `evaluation_agent` maps each criterion against patient data using parallel LLM calls → `verdict_agent` synthesizes individual evaluations into final verdict → `sanitize_output()` redacts PHI → results displayed in Streamlit with expandable reasoning sections.

## Deployment Path

Local development uses `streamlit run app.py` with environment variables for Anthropic API keys and Langfuse credentials. Production deployment targets containerized environment with `Dockerfile` containing Python dependencies, health check endpoints, and resource limits. CI/CD pipeline includes unit tests for guardrails, integration tests for agent chains, and PHI detection validation using synthetic medical data.

## Tradeoffs & Known Limitations

**Model Selection**: Claude Haiku for criteria extraction sacrifices some accuracy for speed—complex trial protocols may require Sonnet. **PHI Detection**: Regex-based approach misses contextual PHI and may over-redact legitimate medical terms. **Evaluation Depth**: Individual criterion evaluation doesn't capture interdependencies between criteria that human coordinators consider. **Scalability**: Sequential agent execution limits throughput—parallel evaluation of independent criteria would improve performance. **Medical Accuracy**: No integration with medical ontologies (SNOMED, ICD-10) for standardized terminology matching.