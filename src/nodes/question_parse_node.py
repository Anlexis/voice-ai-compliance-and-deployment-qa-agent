"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — QuestionParseNode
# Inner workflow node 1: parse the user's voice-AI compliance question,
# extract the structured intent and question type for downstream routing.
#
# Input state keys:
#   validated_input: str — cleaned user question (from outer PreProcessNode)
#
# Output state keys (partial dict):
#   parsed_question:  str — normalised question text
#   question_type:    str — category: "legal_compliance" | "technical_config" |
#                           "accessibility" | "appi" | "general"
#   status:           AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
#   error_log:        list[str] — appended errors (ERROR only)
#
# Security notes:
#   Audit: emit_trace_event on success and error paths.
#   Trust: ANONYMOUS — inner domain node; the outer gate owns the caller check.

import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# ── Question-type keyword patterns (module-level) ──────────────────────────────

_APPI_PATTERN = re.compile(
    r"(?i)(appi|個人情報保護|個人情報|音声録音|録音|同意|保持期間|削除権|consent|retention|deletion)",
)

_ACCESSIBILITY_PATTERN = re.compile(
    r"(?i)(バリアフリー|障害者|アクセシビリティ|accessibility|barrier.free|障害者差別解消|"
    r"手話|字幕|音声ガイド|多言語|language\s+support|multilingual)",
)

_TECHNICAL_PATTERN = re.compile(
    r"(?i)(設定|configuration|deployment|デプロイ|インストール|install|endpoint|API|"
    r"latency|レイテンシ|モデル|model|アーキテクチャ|architecture|testing|テスト|"
    r"accuracy|精度|dialect|方言)",
)

_LEGAL_PATTERN = re.compile(
    r"(?i)(法律|規制|コンプライアンス|法令|compliance|regulation|legal|"
    r"労働法|育成就労|勤労|workforce|派遣|義務|obligation|mandator)",
)


def _classify_question_type(text: str) -> str:
    """Classify the question into a category based on keyword matching.

    Priority order: appi > accessibility > legal_compliance > technical_config > general.
    """
    if _APPI_PATTERN.search(text):
        return "appi"
    if _ACCESSIBILITY_PATTERN.search(text):
        return "accessibility"
    if _LEGAL_PATTERN.search(text):
        return "legal_compliance"
    if _TECHNICAL_PATTERN.search(text):
        return "technical_config"
    return "general"


class QuestionParseNode(FunctionNode):
    """Parse the voice-AI compliance question and classify its type.

    First inner-graph node. Reads validated_input (set by outer PreProcessNode),
    normalises the text, and classifies the question type to guide downstream
    retrieval and annotation nodes.

    Output (partial dict — only changed keys):
        parsed_question, question_type, status, error_log.
    """

    # Inner domain node — ANONYMOUS; the outer gate owns the caller check.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error_log: List[str] = list(state.get("error_log") or [])
        # Inner graph is invoked with user_input=extracted_text from extract_input();
        # validated_input is set in the OUTER state but the inner graph starts fresh
        # with user_input. Fall back to user_input so the pipeline always has input.
        validated_input: str = state.get("validated_input") or state.get("user_input", "")

        if not validated_input or not isinstance(validated_input, str):
            emit_trace_event(
                "question_parse_error",
                {"reason": "validated_input is absent or not a string"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": error_log
                + ["QuestionParseNode: validated_input is absent or not a string; " "PreProcessNode must run first"],
            }

        # Normalise: strip surrounding whitespace, collapse multiple spaces
        parsed = re.sub(r"\s+", " ", validated_input.strip())
        question_type = _classify_question_type(parsed)

        emit_trace_event(
            "question_parsed",
            {
                "question_type": question_type,
                "input_length": len(parsed),
            },
            state,
        )

        return {
            "parsed_question": parsed,
            "question_type": question_type,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }
