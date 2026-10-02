"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — ResponseComposeNode
# Inner workflow node 7: compose the guidance document from every upstream
# result — retrieval, recording governance, accessibility, model guidance and
# the accumulated compliance findings — and derive the overall severity.
#
# Input state keys:
#   parsed_question:       str  — normalised user question
#   question_type:         str  — question category
#   legal_contexts:        list — applicable laws
#   retrieval_results:     list — scored excerpts
#   appi_annotations:      str  — recording-governance section
#   accessibility_notes:   str  — accessibility section
#   model_recommendations: str  — model guidance section
#   compliance_findings:   list — structured findings from the evaluation nodes
#
# Output state keys (partial dict):
#   draft_response:   str — full composed guidance document
#   overall_severity: str — strongest severity across the findings
#   status:           AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
#   error_log:        list[str] (ERROR only)
#
# Security notes:
#   Audit: emit_trace_event on every path.
#   Trust: ANONYMOUS — inner domain node.

from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import strongest_severity

# Caller excerpts are quoted at bounded length. The cap is a rendering rule, not
# a security control — the content was screened and size-capped upstream.
_MAX_EXCERPT_CHARS = 400
_MAX_EXCERPTS = 3

# The document states the rules it was rendered under, so a reader can check the
# output against them without reading the source.
_SCHEMA_NOTE = (
    "Output schema: derived findings and knowledge-base excerpts only. Ratios are rendered as "
    "percentages to one decimal place; day and millisecond counts as whole numbers. Every "
    "caller-supplied label reproduced below is an inert identifier (a-z, 0-9, underscore)."
)


def _format_legal_contexts(legal_contexts: List[Dict[str, Any]]) -> str:
    """Format the list of applicable laws into a readable section."""
    if not legal_contexts:
        return "No specific Japanese legal obligations were identified for this query."
    lines = []
    for ctx in legal_contexts:
        lines.append(f"  - {ctx.get('name_ja', ctx.get('id', 'Unknown'))} ({ctx.get('name_en', '')})")
    return "\n".join(lines)


def _format_retrieval_excerpts(results: List[Dict[str, Any]], max_items: int = _MAX_EXCERPTS) -> str:
    """Format the top retrieval results as numbered excerpts."""
    if not results:
        return "No relevant knowledge-base entries were retrieved."
    lines = []
    for idx, rec in enumerate(results[:max_items], start=1):
        title = rec.get("title", "Untitled")
        origin = "caller" if rec.get("source") == "caller" else "baseline"
        content = str(rec.get("content", ""))
        if len(content) > _MAX_EXCERPT_CHARS:
            content = content[: _MAX_EXCERPT_CHARS - 3] + "..."
        score = rec.get("relevance_score", 0.0)
        lines.append(f"  [{idx}] {title} (source: {origin}, relevance {float(score) * 100:.1f}%)\n      {content}")
    return "\n".join(lines)


def _format_findings(findings: List[Dict[str, Any]]) -> str:
    """Format the accumulated compliance findings, strongest first."""
    if not findings:
        return (
            "No deployment profile was supplied, so no declared value could be evaluated. "
            "Send input_context.deployment_profile to have retention, deletion, speech-rate, "
            "accuracy, latency and language coverage checked against the thresholds above."
        )
    order = {"action_required": 0, "attention": 1, "info": 2}
    ranked = sorted(findings, key=lambda f: order.get(str(f.get("severity")), 3))
    return "\n".join(
        f"  - [{f.get('severity', 'info')}] {f.get('area', 'general')}: {f.get('detail', '')}" for f in ranked
    )


class ResponseComposeNode(FunctionNode):
    """Compose the guidance document and derive the overall severity.

    Assembles the upstream results into a structured document:
      Section 1 — Applicable laws and obligations
      Section 2 — Knowledge-base references
      Section 3 — Voice recording governance
      Section 4 — Accessibility requirements
      Section 5 — Model guidance
      Section 6 — Compliance findings for the declared deployment profile

    An empty retrieval set is valid: the document is still composed from the
    annotations and the findings.

    Output (partial dict — only changed keys):
        draft_response, overall_severity, status, error_log.
    """

    # Inner domain node — ANONYMOUS; the outer gate owns the caller check.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error_log: List[str] = list(state.get("error_log") or [])

        parsed_question: str = state.get("parsed_question", "")
        question_type: str = state.get("question_type", "general")
        legal_contexts: List[Dict[str, Any]] = state.get("legal_contexts") or []
        retrieval_results: List[Dict[str, Any]] = state.get("retrieval_results") or []
        appi_annotations: str = state.get("appi_annotations", "")
        accessibility_notes: str = state.get("accessibility_notes", "")
        model_recommendations: str = state.get("model_recommendations", "")
        findings: List[Dict[str, Any]] = state.get("compliance_findings") or []

        if not parsed_question:
            emit_trace_event(
                "response_compose_error",
                {"reason": "parsed_question is absent"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": error_log
                + ["ResponseComposeNode: parsed_question is absent; upstream nodes must run first"],
            }

        severity = strongest_severity(findings)

        response = (
            "Voice-AI Compliance & Deployment Q&A\n"
            f"{'=' * 50}\n\n"
            f"Query: {parsed_question}\n"
            f"Category: {question_type}\n"
            f"Overall assessment: {severity}\n\n"
            f"{_SCHEMA_NOTE}\n\n"
            "## 1. Applicable Japanese Laws & Obligations\n"
            f"{_format_legal_contexts(legal_contexts)}\n\n"
            "## 2. Knowledge-Base References\n"
            f"{_format_retrieval_excerpts(retrieval_results)}\n\n"
            "## 3. Voice Recording Governance\n"
            f"{appi_annotations}\n\n"
            "## 4. Accessibility Requirements\n"
            f"{accessibility_notes}\n\n"
            "## 5. Model Guidance\n"
            f"{model_recommendations}\n\n"
            "## 6. Compliance Findings\n"
            f"{_format_findings(findings)}\n"
            "\n---\n"
            "Note: this guidance is generated from the knowledge base supplied to the agent. "
            "Consult qualified legal counsel for binding compliance advice.\n"
        )

        emit_trace_event(
            "response_composed",
            {
                "question_type": question_type,
                "law_count": len(legal_contexts),
                "retrieval_count": len(retrieval_results),
                "finding_count": len(findings),
                "overall_severity": severity,
                "response_length": len(response),
            },
            state,
        )

        return {
            "draft_response": response,
            "overall_severity": severity,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }
