"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — APPIVoiceAnnotateNode
# Inner workflow node 4: voice-recording governance under the 2026 personal
# information protection amendments.
#
# Two modes, and the difference is the caller's data:
#   - No deployment profile → the general obligation checklist.
#   - Deployment profile present → the checklist PLUS an evaluation of the
#     declared retention period and deletion service level against the
#     statutory ceilings, with the shortfall computed.
#
# Input state keys:
#   legal_contexts:   list[dict] — identified applicable laws
#   question_type:    str        — question category
#   caller_profile:   dict       — validated deployment parameters
#
# Output state keys (partial dict):
#   appi_annotations:    str  — obligations and, where evaluable, findings
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
from src.schemas.state import SEVERITY_ACTION_REQUIRED, SEVERITY_ATTENTION, SEVERITY_INFO

# ── Statutory ceilings evaluated against the caller's declared profile ────────

# Default maximum retention period for interaction recordings, in days.
_MAX_RETENTION_DAYS = 90.0

# A retention period beyond this is treated as a documented long-term hold
# rather than an oversight — still reportable, but as attention, not action.
_RETENTION_ATTENTION_MARGIN_DAYS = 30.0

# Deletion requests must complete within this many business days.
_MAX_DELETION_SLA_DAYS = 30.0

_CHECKLIST = (
    "Voice recording governance — obligations under the 2026 amendments:\n"
    "  - 事前同意取得: obtain explicit prior consent before recording\n"
    "  - 目的明示: disclose the recording purpose at session start\n"
    "  - 保持期間設定: set and enforce a retention period (90 days by default)\n"
    "  - 削除権対応: implement a deletion-request workflow (30 business days)\n"
    "  - バイオメトリック判定: classify recordings for biometric identifier content\n"
    "  - オプトアウト機能: provide an accessible opt-out at any point\n"
    "  - 第三者提供制限: restrict third-party disclosure without further consent\n"
)

_NOT_APPLICABLE = (
    "Voice recording governance: no specific recording obligations were identified for this "
    "query. The general principles (purpose limitation, accuracy, security safeguards) still apply."
)


class APPIVoiceAnnotateNode(FunctionNode):
    """Evaluate voice-recording governance obligations for the query.

    Emits the obligation checklist when recording rules are in scope, and — when
    the caller declared a retention period or a deletion service level —
    evaluates those declared values against the statutory ceilings and records
    a finding per breach.

    Output (partial dict — only changed keys):
        appi_annotations, compliance_findings, status, error_log.
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
        in_scope = "APPI_2026" in law_ids or question_type == "appi"

        sections: List[str] = [_CHECKLIST if in_scope else _NOT_APPLICABLE]
        new_findings: List[Dict[str, Any]] = []

        # ── Declared retention period ─────────────────────────────────────────
        retention = profile.get("recording_retention_days")
        if retention is not None:
            overrun = retention - _MAX_RETENTION_DAYS
            if overrun > 0:
                severity = (
                    SEVERITY_ATTENTION if overrun <= _RETENTION_ATTENTION_MARGIN_DAYS else SEVERITY_ACTION_REQUIRED
                )
                detail = (
                    f"Declared retention period is {retention:.0f} days, "
                    f"{overrun:.0f} days beyond the {_MAX_RETENTION_DAYS:.0f}-day default ceiling. "
                    "Document the extension basis or shorten the period."
                )
                new_findings.append(
                    {
                        "area": "recording_retention",
                        "code": "retention_exceeds_default",
                        "severity": severity,
                        "detail": detail,
                    }
                )
            else:
                new_findings.append(
                    {
                        "area": "recording_retention",
                        "code": "retention_within_default",
                        "severity": SEVERITY_INFO,
                        "detail": (
                            f"Declared retention period is {retention:.0f} days, within the "
                            f"{_MAX_RETENTION_DAYS:.0f}-day default ceiling."
                        ),
                    }
                )

        # ── Declared deletion service level ───────────────────────────────────
        deletion_sla = profile.get("deletion_sla_days")
        if deletion_sla is not None:
            overrun = deletion_sla - _MAX_DELETION_SLA_DAYS
            if overrun > 0:
                new_findings.append(
                    {
                        "area": "deletion_workflow",
                        "code": "deletion_sla_exceeds_limit",
                        "severity": SEVERITY_ACTION_REQUIRED,
                        "detail": (
                            f"Declared deletion service level is {deletion_sla:.0f} business days, "
                            f"{overrun:.0f} beyond the {_MAX_DELETION_SLA_DAYS:.0f}-day limit."
                        ),
                    }
                )
            else:
                new_findings.append(
                    {
                        "area": "deletion_workflow",
                        "code": "deletion_sla_within_limit",
                        "severity": SEVERITY_INFO,
                        "detail": (
                            f"Declared deletion service level is {deletion_sla:.0f} business days, "
                            f"within the {_MAX_DELETION_SLA_DAYS:.0f}-day limit."
                        ),
                    }
                )

        if new_findings:
            sections.append(
                "\nEvaluation of the declared deployment profile:\n"
                + "\n".join(f"  - [{f['severity']}] {f['detail']}" for f in new_findings)
            )

        emit_trace_event(
            "appi_annotation_evaluated",
            {
                "in_scope": in_scope,
                "law_ids": sorted(law_ids),
                "question_type": question_type,
                "finding_codes": [f["code"] for f in new_findings],
            },
            state,
        )

        return {
            "appi_annotations": "\n".join(sections),
            "compliance_findings": findings + new_findings,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }
