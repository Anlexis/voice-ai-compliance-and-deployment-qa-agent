"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — ModelRecommendNode
# Inner workflow node 6: voice-model guidance for Japanese enterprise
# deployment, plus an evaluation of the caller's declared model performance
# against the thresholds the applicable rules imply.
#
# Input state keys:
#   question_type:   str        — question category
#   legal_contexts:  list[dict] — applicable laws (multilingual requirement)
#   caller_profile:  dict       — validated deployment parameters
#
# Output state keys (partial dict):
#   model_recommendations: str  — guidance and, where evaluable, findings
#   compliance_findings:   list — accumulated structured findings
#   status:                AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
#   error_log:             list[str] (ERROR only)
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

# ── Thresholds evaluated against the caller's declared profile ────────────────

# Minimum per-language intent recognition accuracy for workforce deployments.
_MIN_INTENT_ACCURACY = 0.85

# Conversational response budget, in milliseconds.
_TARGET_LATENCY_MS = 300.0

# Beyond this the interaction stops feeling conversational rather than merely
# being slower than target, which is why the two cases carry different weight.
_LATENCY_ATTENTION_CEILING_MS = 600.0

# Languages a workforce deployment is expected to cover.
_REQUIRED_LANGUAGES = ("japanese", "mandarin", "vietnamese", "indonesian", "tagalog")

_CATALOGUE = (
    "Open-source voice models in common Japanese enterprise use:\n"
    "\n"
    "1. VoxCPM (OpenBMB)\n"
    "   License: Apache 2.0 | Languages: 100+ including Japanese\n"
    "   Strengths: multi-language accuracy, active maintenance, deployment docs\n"
    "   Trade-offs: higher GPU memory footprint (8GB+ recommended)\n"
    "   Use when: broad multilingual coverage is required\n"
    "\n"
    "2. MOSS-TTS (Fudan University)\n"
    "   License: MIT | Languages: Japanese-optimised plus English\n"
    "   Strengths: Japanese prosody, low-latency synthesis, honorific model\n"
    "   Trade-offs: synthesis only; recognition needs a separate model\n"
    "   Use when: Japanese synthesis quality and latency dominate\n"
    "\n"
    "3. VibeVoice\n"
    "   License: check the current terms | Languages: 50+\n"
    "   Strengths: streaming synthesis, dialect adaptation\n"
    "   Trade-offs: newest entrant; production maturity still being established\n"
    "   Use when: evaluating streaming capability at pilot stage\n"
    "\n"
    "Evaluation checklist for Japanese enterprise:\n"
    "  - word error rate on the CSJ corpus at or below 8%\n"
    "  - real-time factor at or below 0.3 on the target hardware\n"
    "  - honorific (keigo) recognition accuracy at or above 90%\n"
    "  - regional dialect coverage (Kansai / Tohoku / Kyushu)\n"
    "  - license compatibility (Apache 2.0 / MIT preferred)\n"
)

_MULTILINGUAL_ADDITION = (
    "\nMultilingual priority for workforce deployments:\n"
    "  Priority languages: Japanese, Mandarin, Vietnamese, Indonesian, Tagalog\n"
    "  VoxCPM is the usual primary choice on coverage breadth\n"
    "  Per-language intent recognition target: 85% or better\n"
)


class ModelRecommendNode(FunctionNode):
    """Recommend voice models and evaluate the caller's declared performance.

    Always emits the model catalogue, adds multilingual guidance where the
    workforce context applies, and — where the caller declared accuracy,
    latency or language coverage — evaluates each against its threshold.

    Output (partial dict — only changed keys):
        model_recommendations, compliance_findings, status, error_log.
    """

    # Inner domain node — ANONYMOUS; the outer gate owns the caller check.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(  # noqa: C901
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
        multilingual = "IKUSEI_SHURO_2028" in law_ids or question_type == "general"

        recommendations = _CATALOGUE
        if multilingual:
            recommendations += _MULTILINGUAL_ADDITION

        new_findings: List[Dict[str, Any]] = []

        # ── Declared intent recognition accuracy ──────────────────────────────
        accuracy = profile.get("intent_accuracy")
        if accuracy is not None:
            shortfall = _MIN_INTENT_ACCURACY - accuracy
            if shortfall > 0:
                new_findings.append(
                    {
                        "area": "model_accuracy",
                        "code": "intent_accuracy_below_target",
                        "severity": SEVERITY_ACTION_REQUIRED,
                        "detail": (
                            f"Declared intent recognition accuracy is {accuracy * 100:.1f}%, "
                            f"{shortfall * 100:.1f} percentage points below the "
                            f"{_MIN_INTENT_ACCURACY * 100:.1f}% target."
                        ),
                    }
                )
            else:
                new_findings.append(
                    {
                        "area": "model_accuracy",
                        "code": "intent_accuracy_meets_target",
                        "severity": SEVERITY_INFO,
                        "detail": (
                            f"Declared intent recognition accuracy is {accuracy * 100:.1f}%, at or "
                            f"above the {_MIN_INTENT_ACCURACY * 100:.1f}% target."
                        ),
                    }
                )

        # ── Declared response latency ─────────────────────────────────────────
        latency = profile.get("response_latency_ms")
        if latency is not None:
            overrun = latency - _TARGET_LATENCY_MS
            if overrun > 0:
                severity = SEVERITY_ATTENTION if latency <= _LATENCY_ATTENTION_CEILING_MS else SEVERITY_ACTION_REQUIRED
                new_findings.append(
                    {
                        "area": "response_latency",
                        "code": "latency_above_target",
                        "severity": severity,
                        "detail": (
                            f"Declared response latency is {latency:.0f}ms, {overrun:.0f}ms above "
                            f"the {_TARGET_LATENCY_MS:.0f}ms conversational target."
                        ),
                    }
                )
            else:
                new_findings.append(
                    {
                        "area": "response_latency",
                        "code": "latency_within_target",
                        "severity": SEVERITY_INFO,
                        "detail": (
                            f"Declared response latency is {latency:.0f}ms, within the "
                            f"{_TARGET_LATENCY_MS:.0f}ms conversational target."
                        ),
                    }
                )

        # ── Declared language coverage ────────────────────────────────────────
        languages = profile.get("supported_languages")
        if languages is not None:
            declared = {str(entry) for entry in languages}
            missing = [name for name in _REQUIRED_LANGUAGES if name not in declared]
            if missing:
                new_findings.append(
                    {
                        "area": "language_coverage",
                        "code": "language_coverage_incomplete",
                        "severity": SEVERITY_ACTION_REQUIRED if multilingual else SEVERITY_ATTENTION,
                        "detail": (
                            f"Declared language coverage is missing {len(missing)} of the "
                            f"{len(_REQUIRED_LANGUAGES)} expected languages: " + ", ".join(missing) + "."
                        ),
                    }
                )
            else:
                new_findings.append(
                    {
                        "area": "language_coverage",
                        "code": "language_coverage_complete",
                        "severity": SEVERITY_INFO,
                        "detail": (
                            f"Declared language coverage includes all {len(_REQUIRED_LANGUAGES)} " "expected languages."
                        ),
                    }
                )

        if new_findings:
            recommendations += (
                "\nEvaluation of the declared deployment profile:\n"
                + "\n".join(f"  - [{f['severity']}] {f['detail']}" for f in new_findings)
                + "\n"
            )

        emit_trace_event(
            "model_recommendations_generated",
            {
                "multilingual_guidance": multilingual,
                "question_type": question_type,
                "finding_codes": [f["code"] for f in new_findings],
            },
            state,
        )

        return {
            "model_recommendations": recommendations,
            "compliance_findings": findings + new_findings,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }
