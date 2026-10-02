"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — AccessibilityCheckFormatNode
# Inner workflow node 5: accessibility requirements for voice interfaces under
# the 2024 barrier-free amendment and the anti-discrimination act.
#
# When the caller declares a speech-rate range, the node evaluates it against
# the mandated 0.5x-2.0x control range and reports the shortfall at each end.
#
# Input state keys:
#   legal_contexts:   list[dict] — identified applicable laws
#   question_type:    str        — question category
#   caller_profile:   dict       — validated deployment parameters
#
# Output state keys (partial dict):
#   accessibility_notes: str  — requirements and, where evaluable, findings
#   compliance_findings: list — accumulated structured findings
#   status:              AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
#   error_log:           list[str] (ERROR only)
#
# Security notes:
#   Audit: emit_trace_event on every path.
#   Trust: ANONYMOUS — inner domain node.

from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import SEVERITY_ACTION_REQUIRED, SEVERITY_INFO

# ── Mandated speech-rate control range ────────────────────────────────────────
_REQUIRED_RATE_MIN = 0.5
_REQUIRED_RATE_MAX = 2.0

_REQUIREMENTS = (
    "Voice AI accessibility requirements (2024 barrier-free amendment / "
    "anti-discrimination act):\n"
    "  1. 字幕・キャプション: real-time speech-to-text captions for every voice\n"
    "     interaction, usable by hearing-impaired callers.\n"
    "  2. 手話アバター: public-service voice interfaces must offer a Japanese\n"
    "     Sign Language avatar channel as an alternative.\n"
    "  3. 調整可能な発話速度: speech-rate control across 0.5x-2.0x, default 1.0x,\n"
    "     with the preference persisted across sessions.\n"
    "  4. 方言認識: dialect-aware recognition covering at least the Kansai,\n"
    "     Tohoku and Kyushu regional accents.\n"
    "  5. 合理的配慮: a fallback to a human operator or a text channel whenever\n"
    "     the voice interface fails for a user.\n"
    "  6. 差別禁止: service quality must not degrade for users with speech\n"
    "     impairments or non-standard pronunciation.\n"
)

_NOT_APPLICABLE = (
    "Accessibility: no specific barrier-free or anti-discrimination obligations were identified "
    "for this query. Voice interface components should still follow WCAG 2.1 AA."
)


class AccessibilityCheckFormatNode(FunctionNode):
    """Summarise accessibility requirements and evaluate the declared profile.

    Applies when the barrier-free or anti-discrimination contexts are in scope,
    or when the question itself is an accessibility question. A declared
    speech-rate range is evaluated against the mandated range at both ends.

    Output (partial dict — only changed keys):
        accessibility_notes, compliance_findings, status, error_log.
    """

    # Inner domain node — ANONYMOUS; the outer gate owns the caller check.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error_log: List[str] = list(state.get("error_log") or [])
        findings: List[Dict[str, Any]] = list(state.get("compliance_findings") or [])
        legal_contexts: List[Dict[str, Any]] = state.get("legal_contexts") or []
        question_type: str = state.get("question_type", "general")
        profile: Dict[str, Any] = state.get("caller_profile") or {}

        law_ids = {ctx.get("id", "") for ctx in legal_contexts}
        in_scope = "BARRIER_FREE_2024" in law_ids or "DISABILITY_ACT" in law_ids or question_type == "accessibility"

        sections: List[str] = [_REQUIREMENTS if in_scope else _NOT_APPLICABLE]
        new_findings: List[Dict[str, Any]] = []

        rate_min = profile.get("speech_rate_min")
        rate_max = profile.get("speech_rate_max")
        if rate_min is not None or rate_max is not None:
            shortfalls: List[str] = []
            if rate_min is not None and rate_min > _REQUIRED_RATE_MIN:
                shortfalls.append(f"slowest rate {rate_min:.2f}x does not reach the required {_REQUIRED_RATE_MIN:.2f}x")
            if rate_max is not None and rate_max < _REQUIRED_RATE_MAX:
                shortfalls.append(f"fastest rate {rate_max:.2f}x does not reach the required {_REQUIRED_RATE_MAX:.2f}x")
            if shortfalls:
                new_findings.append(
                    {
                        "area": "speech_rate_control",
                        "code": "speech_rate_range_insufficient",
                        "severity": SEVERITY_ACTION_REQUIRED,
                        "detail": (
                            "Declared speech-rate control does not cover the mandated "
                            f"{_REQUIRED_RATE_MIN:.2f}x-{_REQUIRED_RATE_MAX:.2f}x range: " + "; ".join(shortfalls) + "."
                        ),
                    }
                )
            else:
                new_findings.append(
                    {
                        "area": "speech_rate_control",
                        "code": "speech_rate_range_sufficient",
                        "severity": SEVERITY_INFO,
                        "detail": (
                            "Declared speech-rate control covers the mandated "
                            f"{_REQUIRED_RATE_MIN:.2f}x-{_REQUIRED_RATE_MAX:.2f}x range."
                        ),
                    }
                )

        if new_findings:
            sections.append(
                "\nEvaluation of the declared deployment profile:\n"
                + "\n".join(f"  - [{f['severity']}] {f['detail']}" for f in new_findings)
            )

        emit_trace_event(
            "accessibility_check_evaluated",
            {
                "in_scope": in_scope,
                "question_type": question_type,
                "finding_codes": [f["code"] for f in new_findings],
            },
            state,
        )

        return {
            "accessibility_notes": "\n".join(sections),
            "compliance_findings": findings + new_findings,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }
