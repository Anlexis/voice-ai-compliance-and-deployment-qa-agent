"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — LegalContextIdentifyNode
# Inner workflow node 2: identify applicable Japanese laws and
# regulations for the parsed voice-AI question, building a legal-context list
# used by downstream retrieval and annotation nodes.
#
# Input state keys:
#   parsed_question: str  — normalised question text
#   question_type:   str  — question category from QuestionParseNode
#
# Output state keys (partial dict):
#   legal_contexts: list[dict]  — list of applicable legal/regulatory references
#   status:         AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
#   error_log:      list[str] (ERROR only)
#
# Security notes:
#   Audit: emit_trace_event on every path.
#   Trust: ANONYMOUS — inner domain node.

import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# ── Law/regulation catalogue (module-level) ───────────────────────────────────
# Each entry: {id, name_ja, name_en, year, scope, triggers}
# 'triggers' is a compiled regex — if it matches, this law is applicable.

_LAW_CATALOGUE: List[Dict[str, Any]] = [
    {
        "id": "APPI_2026",
        "name_ja": "改正個人情報保護法 2026年施行",
        "name_en": "APPI Amendment 2026 (Voice Recording Obligations)",
        "year": 2026,
        "scope": "voice_recording_consent_retention_deletion",
        "triggers": re.compile(
            r"(?i)(appi|個人情報保護|音声録音|録音|同意|保持期間|削除権|consent|retention|deletion)",
        ),
    },
    {
        "id": "BARRIER_FREE_2024",
        "name_ja": "バリアフリー法 2024年改正（音声 AI 対応義務化）",
        "name_en": "Barrier-Free Act 2024 Amendment (Voice AI Accessibility Mandate)",
        "year": 2024,
        "scope": "voice_accessibility_mandate",
        "triggers": re.compile(
            r"(?i)(バリアフリー|障害者差別解消|アクセシビリティ|accessibility|barrier.free|" r"手話|字幕|音声ガイド)",
        ),
    },
    {
        "id": "DISABILITY_ACT",
        "name_ja": "障害者差別解消法",
        "name_en": "Act for Eliminating Discrimination against Persons with Disabilities",
        "year": 2016,
        "scope": "disability_non_discrimination_voice_interface",
        "triggers": re.compile(
            r"(?i)(障害者差別解消|障害者|合理的配慮|reasonable\s+accommodation|disability)",
        ),
    },
    {
        "id": "IKUSEI_SHURO_2028",
        "name_ja": "育成就労制度 2028年施行（外国人労働者向け音声 AI）",
        "name_en": "Ikusei-Shuro (Skilled Labour Development) 2028 — multilingual voice AI",
        "year": 2028,
        "scope": "multilingual_voice_ai_workforce",
        "triggers": re.compile(
            r"(?i)(育成就労|外国人労働|多言語|multilingual|language\s+support|dialect|方言|" r"workforce|労働者)",
        ),
    },
]


class LegalContextIdentifyNode(FunctionNode):
    """Identify applicable Japanese laws for the voice-AI compliance question.

    Scans parsed_question and question_type against the law catalogue and returns
    a list of applicable legal contexts. An empty list is valid (no specific law
    applies) and results in SUCCESS with a 'general' guidance path.

    Output (partial dict — only changed keys):
        legal_contexts, status, error_log.
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

        if not parsed_question:
            emit_trace_event(
                "legal_context_identify_error",
                {"reason": "parsed_question is absent"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": error_log
                + ["LegalContextIdentifyNode: parsed_question is absent; " "QuestionParseNode must run first"],
            }

        # Scan text against catalogue
        applicable: List[Dict[str, Any]] = []
        scan_text = f"{parsed_question} {question_type}"

        for law in _LAW_CATALOGUE:
            if law["triggers"].search(scan_text):
                applicable.append(
                    {
                        "id": law["id"],
                        "name_ja": law["name_ja"],
                        "name_en": law["name_en"],
                        "year": law["year"],
                        "scope": law["scope"],
                    }
                )

        emit_trace_event(
            "legal_contexts_identified",
            {
                "applicable_law_count": len(applicable),
                "law_ids": [law["id"] for law in applicable],
                "question_type": question_type,
            },
            state,
        )

        return {
            "legal_contexts": applicable,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }
