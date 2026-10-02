"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption. Extend AgentState with agent-specific fields only. Do NOT add
# credentials, secrets, or Pydantic models.

import math
from typing import Any, Dict, List, Optional

from framework.schemas.agent_state import AgentState

# Caller-supplied strings that render into the guidance document are locked to
# this alphabet. Free text in a rendered label is caller-controlled output
# injection; an identifier cannot carry a directive, a marker, or a line break.
IDENTIFIER_PATTERN = r"^[a-z0-9_]{1,32}$"

# Severity of a single compliance finding, weakest first. The overall severity
# of an answer is the strongest finding it carries.
SEVERITY_INFO = "info"
SEVERITY_ATTENTION = "attention"
SEVERITY_ACTION_REQUIRED = "action_required"
SEVERITY_ORDER = (SEVERITY_INFO, SEVERITY_ATTENTION, SEVERITY_ACTION_REQUIRED)


def strongest_severity(findings: Any) -> str:
    """Return the strongest severity present in a findings list."""
    strongest = SEVERITY_INFO
    if not isinstance(findings, list):
        return strongest
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        severity = finding.get("severity")
        if severity in SEVERITY_ORDER and SEVERITY_ORDER.index(severity) > SEVERITY_ORDER.index(strongest):
            strongest = severity
    return strongest


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse a caller-controlled number: FINITE float within [lo, hi], else None.

    Rejects bools, non-numerics, and — the reason this helper exists — every
    non-finite value. ``float()`` parses ``"NaN"`` and ``"Infinity"`` happily,
    Python's ``json`` accepts bare ``NaN`` in a request body, and IEEE NaN
    compares False against everything. A NaN retention period would therefore
    pass every ``>`` threshold check and be reported as compliant: the failure
    is silent and it is on exactly the decision this agent exists to make.

    Callers learn WHICH field was rejected, never what value they sent.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class State(AgentState):
    """CMN-C2-659 state — Voice-AI Compliance & Deployment Q&A Agent.

    Field lifecycle:
      validated_input   — set by PreProcessNode (outer); read by inner graph nodes
      enriched_context  — set by PreProcessNode (outer); metadata for inner nodes
      caller_profile    — set by PreProcessNode (outer); validated deployment
                          parameters from input_context, carried across the
                          outer→inner boundary by the context bridge
      caller_documents  — set by PreProcessNode (outer); validated caller-supplied
                          knowledge-base entries, same carriage
      parsed_question   — set by QuestionParseNode; structured query representation
      question_type     — set by QuestionParseNode; one of:
                          "legal_compliance" | "technical_config" |
                          "accessibility" | "appi" | "general"
      legal_contexts    — set by LegalContextIdentifyNode; list of applicable laws
      retrieval_results — set by HybridRetrieveNode; list of knowledge-base excerpts
      appi_annotations  — set by APPIVoiceAnnotateNode; recording-governance findings
      accessibility_notes — set by AccessibilityCheckFormatNode; accessibility findings
      model_recommendations — set by ModelRecommendNode; voice-model guidance
      compliance_findings — set by the evaluation nodes; accumulated structured
                          findings, each {area, code, severity, detail}
      overall_severity  — derived by ResponseComposeNode from compliance_findings
      draft_response    — set by ResponseComposeNode; full composed Q&A text
      secured_response  — set by SecurityGateOutputNode; released output
      result            — set by the outer graph node's merge_output()
      formatted_output  — set by PostProcessNode (outer)

    All fields are Optional to allow incremental state construction through
    the linear pipeline. No Pydantic models, dataclasses, or arbitrary Python
    objects — flat primitives only (str, dict, list with JSON-serialisable values).
    """

    # ── Outer pre_process enrichment ──────────────────────────────────────────
    validated_input: Optional[str]
    enriched_context: Optional[Dict[str, Any]]

    # ── Validated caller data (carried across the outer→inner boundary) ───────
    caller_profile: Optional[Dict[str, Any]]
    caller_documents: Optional[List[Dict[str, Any]]]

    # ── Inner domain workflow ─────────────────────────────────────────────────
    parsed_question: Optional[str]
    question_type: Optional[str]
    legal_contexts: Optional[List[Dict[str, Any]]]
    retrieval_results: Optional[List[Dict[str, Any]]]
    appi_annotations: Optional[str]
    accessibility_notes: Optional[str]
    model_recommendations: Optional[str]
    compliance_findings: Optional[List[Dict[str, Any]]]
    overall_severity: Optional[str]
    draft_response: Optional[str]
    secured_response: Optional[str]

    # ── Outer post_process ────────────────────────────────────────────────────
    result: Optional[str]
    formatted_output: Optional[str]
